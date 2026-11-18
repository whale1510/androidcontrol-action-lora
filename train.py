#!/usr/bin/env python3
"""LoRA SFT of Qwen3-VL on the AndroidControl action data (data/train.jsonl).

Follows LLaMA-Factory's SFT path for Qwen3-VL (template `qwen3_vl_nothink`, finetuning_type lora):
  * text   : the model's own chat template; the answer is followed by "<|im_end|>\\n"
  * labels : everything up to and including "<|im_start|>assistant\\n" is ignored (train_on_prompt
             false), so the loss covers only the answer (thought + action JSON) and the
             end-of-turn token
  * image  : resized once to at most `image_max_pixels` (prompts.resize_image), then the processor
             expands the image token to one token per 32x32 pixels
  * LoRA   : every Linear layer of the language model (lora_target: all); vision tower and
             projector frozen (freeze_vision_tower / freeze_multi_modal_projector true); no new tokens
  * loop   : transformers Trainer, cosine schedule with warmup, bf16

    CUDA_VISIBLE_DEVICES=0 python train.py --config configs/lora_4b.yaml
    ... --set max_steps=30 output_dir=runs/try          # short trial
    ... --set resume=runs/lora_4b/checkpoint-1100       # continue an interrupted run
"""
import argparse
import json
import math
import os

import torch
import yaml
from torch.utils.data import Dataset

import modeling as M
import paths
DEFAULTS = {
    # model
    "model": M.DEFAULT_MODEL,
    "image_max_pixels": M.DEFAULT_MAX_PIXELS,
    "image_min_pixels": M.DEFAULT_MIN_PIXELS,
    # data
    "train_file": "data/train.jsonl",
    "eval_file": "data/val.jsonl",
    "image_root": None,                   # None = data/androidcontrol (or $ANDROIDCONTROL_DIR)
    "max_train_samples": 0,               # 0 = all
    "max_eval_samples": 0,
    "cutoff_len": 4096,                   # examples longer than this are an error, not truncated
    # lora
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target": "all",                 # "all" = every Linear in the language model, or a comma list
    # optimisation
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1.0e-4,
    "num_train_epochs": 1.0,
    "max_steps": -1,                      # > 0 overrides num_train_epochs (counted in optimizer updates)
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
    "seed": 42,
    # bookkeeping
    "output_dir": "runs/lora_4b",
    "logging_steps": 10,
    "save_fraction": 0.1,                 # checkpoint (and eval loss) every this fraction of the run
    "per_device_eval_batch_size": 8,
    "dataloader_num_workers": 8,
    "resume": None,
    "device": None,                       # "cpu" for a code check without a GPU
}


def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    o = ap.parse_args()
    cfg = dict(DEFAULTS)
    if o.config:
        cfg.update(yaml.safe_load(open(o.config)) or {})
    for kv in o.set:
        k, v = kv.split("=", 1)
        if k not in DEFAULTS:
            raise SystemExit(f"unknown setting: {k}")
        cfg[k] = yaml.safe_load(v)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"unknown settings in config: {sorted(unknown)}")
    return cfg


resolve = paths.resolve


class JsonlDataset(Dataset):
    def __init__(self, path, limit=0):
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Rows -> padded batch. Labels are -100 on the prompt and on padding."""

    def __init__(self, processor, image_root, max_pixels, min_pixels, cutoff_len):
        self.proc, self.root = processor, image_root
        self.max_pixels, self.min_pixels, self.cutoff = max_pixels, min_pixels, cutoff_len
        tok = processor.tokenizer
        tok.padding_side = "right"
        self.head = tok(M.ASSISTANT_HEAD, add_special_tokens=False).input_ids

    def answer_start(self, ids):
        n = len(self.head)
        for s in range(len(ids) - n, -1, -1):
            if ids[s:s + n] == self.head:
                return s + n
        raise ValueError("assistant header not found")

    def __call__(self, rows):
        texts, images = [], []
        for r in rows:
            texts.append(M.chat_prompt(self.proc, r) + r["messages"][2]["content"] + M.END_OF_TURN)
            images.append(M.load_image(self.root, r, self.max_pixels, self.min_pixels))
        batch = self.proc(text=texts, images=images, return_tensors="pt", padding=True)
        ids = batch["input_ids"]
        if ids.shape[1] > self.cutoff:
            raise ValueError(f"example of {ids.shape[1]} tokens exceeds cutoff_len {self.cutoff}")
        labels = ids.clone()
        for b in range(ids.shape[0]):
            labels[b, :self.answer_start(ids[b].tolist())] = -100
        labels[batch["attention_mask"] == 0] = -100
        out = dict(batch)
        out["labels"] = labels
        return out


def lora_targets(model, spec):
    """Full names of the Linear layers to adapt: language model only (LLaMA-Factory's
    find_all_linear_modules + patch_target_modules with the vision tower and projector frozen)."""
    leaf = None if spec == "all" else {s.strip() for s in spec.split(",")}
    names = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if "language_model" not in parts or "lm_head" in parts or "visual" in parts:
            continue
        if isinstance(module, torch.nn.Linear) and (leaf is None or parts[-1] in leaf):
            names.append(name)
    if not names:
        raise ValueError("no LoRA target modules found")
    return names


def main():
    cfg = load_cfg()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments, set_seed

    set_seed(cfg["seed"])
    cpu = cfg["device"] == "cpu"
    out_dir = resolve(cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    processor = M.load_processor(cfg["model"], cfg["image_max_pixels"], cfg["image_min_pixels"])

    use_bf16 = bool(cfg["bf16"]) and not cpu
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model"], dtype=torch.bfloat16 if use_bf16 else torch.float32)
    targets = lora_targets(model, cfg["lora_target"])
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], target_modules=targets, bias="none"))
    model.print_trainable_parameters()
    print("LoRA layers:", len(targets), "| kinds:", sorted({t.split('.')[-1] for t in targets}))

    train_ds = JsonlDataset(resolve(cfg["train_file"]), cfg["max_train_samples"])
    eval_ds = JsonlDataset(resolve(cfg["eval_file"]), cfg["max_eval_samples"]) if cfg["eval_file"] else None
    image_root = resolve(cfg["image_root"]) if cfg["image_root"] else paths.androidcontrol_dir()
    collator = Collator(processor, image_root, cfg["image_max_pixels"], cfg["image_min_pixels"],
                        cfg["cutoff_len"])

    per_update = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    updates = cfg["max_steps"] if cfg["max_steps"] > 0 else math.ceil(
        len(train_ds) / per_update * cfg["num_train_epochs"])
    save_steps = max(1, math.ceil(updates * cfg["save_fraction"]))
    print(f"train examples {len(train_ds)} | per update {per_update} | updates {updates} | save every {save_steps}")
    json.dump(cfg, open(os.path.join(out_dir, "train_config.json"), "w"), indent=1)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg["max_steps"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=min(cfg["warmup_steps"], max(1, updates // 10)),
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        bf16=use_bf16,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=save_steps,
        dataloader_num_workers=cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        seed=cfg["seed"],
        use_cpu=cpu,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator)
    result = trainer.train(resume_from_checkpoint=cfg["resume"])
    final = os.path.join(out_dir, "final")
    model.save_pretrained(final)                             # adapter only (LoRA matrices)
    json.dump({"train": result.metrics, "log": trainer.state.log_history},
              open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
    print("saved", final)


if __name__ == "__main__":
    main()
