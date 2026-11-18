"""Model-side helpers shared by train.py and predict.py: processor, chat prompt, image loading."""
import os

from PIL import Image

import prompts as P

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MAX_PIXELS = 768 * 1696       # a 1080x2400 screenshot becomes 736x1696 = 1,219 image tokens
DEFAULT_MIN_PIXELS = 32 * 32 * 4
ASSISTANT_HEAD = "<|im_start|>assistant\n"
END_OF_TURN = "<|im_end|>\n"


def load_processor(model, max_pixels, min_pixels=DEFAULT_MIN_PIXELS):
    """Processor whose own resize limits match prompts.resize_image, so it never resizes again."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model)
    size = dict(processor.image_processor.size)
    size["longest_edge"], size["shortest_edge"] = int(max_pixels), int(min_pixels)
    processor.image_processor.size = {k: v for k, v in size.items() if v is not None}
    return processor


def chat_prompt(processor, row):
    """Prompt text up to and including the assistant header, built by the model's chat template."""
    system, user = row["messages"][0]["content"], row["messages"][1]["content"]
    before, after = user.split(P.IMAGE_TAG)                 # exactly one image per example
    content = ([{"type": "text", "text": before}] if before else []) + [{"type": "image"}]
    if after:
        content.append({"type": "text", "text": after})
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load_image(image_root, row, max_pixels, min_pixels=DEFAULT_MIN_PIXELS):
    with Image.open(os.path.join(image_root, row["images"][0])) as im:
        return P.resize_image(im, max_pixels, min_pixels)
