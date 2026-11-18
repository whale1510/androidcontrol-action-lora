"""Prompt, output format, coordinate and image-size rules for the action LoRA.

Single source of truth: build_sft.py, train.py, predict.py, score.py and the AndroidWorld agent
all import this file, so training and inference see the same text and the same image size.

Action space = AndroidControl's 8 actions with their original keys, plus the terminal `status`
action the AndroidControl paper appends to each episode. Only click and long_press carry
coordinates; they are written as integers 0-999 relative to the screenshot.

Each step is a thought (AndroidControl's step instruction: one sentence saying what to do now)
followed by one action. The history shows the thought and action of every previous step.
"""
import json
import math
import re

IMAGE_TAG = "<image>"          # placeholder in the user text (LLaMA-Factory convention)
COORD_MAX = 999
MAX_HISTORY = 20               # keep the most recent N previous steps in the prompt
PATCH = 32                     # Qwen3-VL: one image token covers 32x32 pixels (patch 16, merge 2)

SYSTEM_PROMPT = """You are a mobile GUI agent operating an Android phone. Given a goal, the history of previous steps, and the current screenshot, decide the next step: write a short thought, then one action.

# Action space
{"action_type":"click","x":<x>,"y":<y>} - tap the point.
{"action_type":"long_press","x":<x>,"y":<y>} - long-press the point.
{"action_type":"scroll","direction":<"up"|"down"|"left"|"right">} - scroll the screen to reveal content in that direction.
{"action_type":"input_text","text":<text>} - type text into the focused field. It does not tap the field or press enter.
{"action_type":"open_app","app_name":<name>} - open the app with that name.
{"action_type":"navigate_back"} - go back to the previous screen.
{"action_type":"navigate_home"} - go to the home screen.
{"action_type":"wait"} - wait for the screen to update.
{"action_type":"status","goal_status":"successful"} - the goal has been completed.

# Coordinates
x and y are integers from 0 to 999, relative to the screenshot: x from the left edge, y from the top edge.

# History
Each previous step is one line: <step number>. <thought> | <action>

# Output
Thought: <one short sentence saying what to do in this step>
Action: <one JSON object from the action space>"""

STATUS_ACTION = {"action_type": "status", "goal_status": "successful"}
STATUS_THOUGHT = "The goal has been completed."
SCROLL_DIRECTIONS = ("up", "down", "left", "right")
# action_type -> required argument keys, in output order
ACTION_KEYS = {
    "click": ("x", "y"),
    "long_press": ("x", "y"),
    "scroll": ("direction",),
    "input_text": ("text",),
    "open_app": ("app_name",),
    "navigate_back": (),
    "navigate_home": (),
    "wait": (),
    "status": ("goal_status",),
}


# ---------------------------------------------------------------------------- coordinates
def to_rel(v, size):
    """Pixel coordinate -> 0..999 (clamped)."""
    if size <= 1:
        return 0
    return max(0, min(COORD_MAX, round(v / (size - 1) * COORD_MAX)))


def to_px(v, size):
    """0..999 -> pixel coordinate (clamped to the screen)."""
    return max(0, min(size - 1, round(v / COORD_MAX * (size - 1))))


def action_to_rel(action, width, height):
    """AndroidControl action (pixel x/y) -> model action (0..999 x/y), keys in canonical order."""
    t = action["action_type"]
    out = {"action_type": t}
    for k in ACTION_KEYS[t]:
        if k == "x":
            out[k] = to_rel(action[k], width)
        elif k == "y":
            out[k] = to_rel(action[k], height)
        else:
            out[k] = action[k]
    return out


def action_to_px(action, width, height):
    """Model action (0..999) -> executable action (pixels)."""
    out = dict(action)
    if "x" in out and "y" in out:
        out["x"], out["y"] = to_px(out["x"], width), to_px(out["y"], height)
    return out


def action_text(action):
    """Canonical one-line JSON, e.g. {"action_type":"click","x":499,"y":944}."""
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------- prompt
def clean_thought(text):
    """One line, single spaces."""
    return " ".join((text or "").split())


def history_line(number, thought, action):
    thought = clean_thought(thought)
    return f"{number}. {thought} | {action_text(action)}" if thought else f"{number}. {action_text(action)}"


def user_text(goal, history):
    """User turn. `history` = previous steps, oldest first, each {"thought": str, "action": model-space dict}.

    Only the last MAX_HISTORY steps are shown; numbering keeps the true step number.
    """
    history = list(history)
    start = max(0, len(history) - MAX_HISTORY)
    if history:
        lines = "\n".join(history_line(start + j + 1, h["thought"], h["action"])
                          for j, h in enumerate(history[start:]))
        hist = f"History:\n{lines}"
    else:
        hist = "History: None"
    return f"Goal: {clean_thought(goal)}\n{hist}\n{IMAGE_TAG}"


def thought_prefix(thought):
    """Start of the answer up to the action; used to force a given thought (low-level evaluation)."""
    return f"Thought: {clean_thought(thought)}\nAction:"


def answer_text(thought, action):
    return f"{thought_prefix(thought)} {action_text(action)}"


def build_messages(goal, history, thought=None, action=None):
    """Chat messages in LLaMA-Factory's sharegpt/openai layout (role/content strings)."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text(goal, history)},
    ]
    if action is not None:
        msgs.append({"role": "assistant", "content": answer_text(thought, action)})
    return msgs


# ---------------------------------------------------------------------------- output parsing
def parse_action(text):
    """Text containing one JSON object -> validated model-space action dict, or None."""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    t = obj.get("action_type")
    if t not in ACTION_KEYS:
        return None
    out = {"action_type": t}
    for k in ACTION_KEYS[t]:
        if k not in obj:
            return None
        v = obj[k]
        if k in ("x", "y"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= COORD_MAX:
                return None
            v = int(round(v))
        elif not isinstance(v, str):
            return None
        if k == "direction" and v not in SCROLL_DIRECTIONS:
            return None
        out[k] = v
    return out


_ACTION_MARK = re.compile(r"(?:^|\n)\s*Action:", re.S)
_THOUGHT_MARK = re.compile(r"^\s*Thought:\s*", re.S)


def parse_output(text):
    """Model answer -> {"thought": str or None, "action": dict or None}.

    The action is read from the text after the last "Action:" line.
    """
    marks = list(_ACTION_MARK.finditer(text))
    if not marks:
        return {"thought": None, "action": parse_action(text)}
    head, tail = text[:marks[-1].start()], text[marks[-1].end():]
    thought = clean_thought(_THOUGHT_MARK.sub("", head, count=1)) or None
    return {"thought": thought, "action": parse_action(tail)}


# ---------------------------------------------------------------------------- image size
def resized_hw(height, width, max_pixels, min_pixels=PATCH * PATCH * 4):
    """Target (height, width): multiples of 32, area within [min_pixels, max_pixels], aspect kept.

    Same rule as Qwen's smart_resize, so the processor does not resize the image again.
    """
    h = max(PATCH, round(height / PATCH) * PATCH)
    w = max(PATCH, round(width / PATCH) * PATCH)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(PATCH, math.floor(height / beta / PATCH) * PATCH)
        w = max(PATCH, math.floor(width / beta / PATCH) * PATCH)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / PATCH) * PATCH
        w = math.ceil(width * beta / PATCH) * PATCH
    return h, w


def resize_image(img, max_pixels, min_pixels=PATCH * PATCH * 4):
    """PIL image -> RGB image at the model input size. Use this in training and at inference."""
    from PIL import Image
    h, w = resized_hw(img.height, img.width, max_pixels, min_pixels)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if (w, h) != img.size:
        img = img.resize((w, h), Image.BICUBIC)
    return img


def image_tokens(height, width, max_pixels):
    h, w = resized_hw(height, width, max_pixels)
    return (h // PATCH) * (w // PATCH)
