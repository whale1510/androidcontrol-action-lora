#!/usr/bin/env python3
"""Generate answers for data/*.jsonl with the base model, or the base model + a LoRA adapter.

Modes:
  high  goal + history + screenshot -> thought + action. This is what the agent does on AndroidWorld.
  low   the reference thought (the step instruction) is given as the start of the answer and the
        model writes only the action. Measures execution when the step is already decided.

Greedy decoding. Re-running with the same --out continues where it stopped.

    python predict.py --data data/val.jsonl --out runs/base/pred_val_high.jsonl                 # untrained base
    python predict.py --data data/val.jsonl --out runs/lora_4b/pred_val_high.jsonl --adapter runs/lora_4b/final
    python predict.py ... --mode low
With --adapter, the model and image size default to the adapter's train_config.json.
"""
import argparse
import json
import os
import time

import torch

import modeling as M
import paths
import prompts as P

resolve = paths.resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", help="LoRA adapter folder (runs/<name>/final or a checkpoint-N)")
    ap.add_argument("--model", help="base model; default: the adapter's, else " + M.DEFAULT_MODEL)
    ap.add_argument("--image_max_pixels", type=int, help="default: the adapter's, else %d" % M.DEFAULT_MAX_PIXELS)
    ap.add_argument("--image_root", help="default: data/androidcontrol (or $ANDROIDCONTROL_DIR)")
    ap.add_argument("--mode", choices=("high", "low"), default="high")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows")
    ap.add_argument("--device", default=None, help='"cpu" for a code check without a GPU')
    o = ap.parse_args()

    trained = {}
    if o.adapter:
        cfg_path = os.path.join(os.path.dirname(resolve(o.adapter).rstrip("/")), "train_config.json")
        if os.path.exists(cfg_path):
            trained = json.load(open(cfg_path))
    model_name = o.model or trained.get("model") or M.DEFAULT_MODEL
    max_pixels = o.image_max_pixels or trained.get("image_max_pixels") or M.DEFAULT_MAX_PIXELS
    min_pixels = trained.get("image_min_pixels") or M.DEFAULT_MIN_PIXELS
    image_root = resolve(o.image_root) if o.image_root else paths.androidcontrol_dir()

    rows = [json.loads(line) for line in open(resolve(o.data), encoding="utf-8")]
    if o.limit:
        rows = rows[:o.limit]
    out_path = resolve(o.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line)
            done.add((d["episode_id"], d["step"]))
    todo = [r for r in rows if (r["meta"]["episode_id"], r["meta"]["step"]) not in done]
    if o.mode == "low":                                   # needs a reference thought to give
        todo = [r for r in todo if P.parse_output(r["messages"][2]["content"])["thought"]]
    print(f"rows {len(rows)} | already done {len(done)} | to do {len(todo)} | mode {o.mode} | "
          f"model {model_name} | adapter {o.adapter} | image_max_pixels {max_pixels}")
    if not todo:
        return

    from transformers import AutoModelForImageTextToText
    cpu = o.device == "cpu"
    device = "cpu" if cpu else "cuda"
    processor = M.load_processor(model_name, max_pixels, min_pixels)
    processor.tokenizer.padding_side = "left"             # batch generation continues from the right edge
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.float32 if cpu else torch.bfloat16)
    if o.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resolve(o.adapter))
    model.to(device).eval()

    t0, n = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as out:
        for s in range(0, len(todo), o.batch_size):
            batch = todo[s:s + o.batch_size]
            prefixes = [P.thought_prefix(P.parse_output(r["messages"][2]["content"])["thought"])
                        if o.mode == "low" else "" for r in batch]
            texts = [M.chat_prompt(processor, r) + pre for r, pre in zip(batch, prefixes)]
            images = [M.load_image(image_root, r, max_pixels, min_pixels) for r in batch]
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=o.max_new_tokens, do_sample=False)
            new = gen[:, inputs["input_ids"].shape[1]:]
            for r, pre, text in zip(batch, prefixes, processor.batch_decode(new, skip_special_tokens=True)):
                out.write(json.dumps({"episode_id": r["meta"]["episode_id"], "step": r["meta"]["step"],
                                      "mode": o.mode, "pred_text": pre + text}, ensure_ascii=False) + "\n")
            out.flush()
            n += len(batch)
            if (s // o.batch_size) % 10 == 0 or n == len(todo):
                dt = time.time() - t0
                print(f"{n}/{len(todo)}  {dt:.0f}s  {n / dt:.2f} rows/s", flush=True)


if __name__ == "__main__":
    main()
