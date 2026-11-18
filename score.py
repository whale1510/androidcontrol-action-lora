#!/usr/bin/env python3
"""Score predictions against data/*.jsonl with AndroidControl's step-accuracy rule. Standard library only.

A step is correct when the action type matches and its argument is right:
  click, long_press   the predicted point lies inside the reference element (meta.target_bbox,
                      the smallest accessibility node containing the recorded point)
  scroll              same direction (reference = the corrected label where one was fixed)
  input_text          same text, ignoring case and extra spaces
  open_app            same app name, ignoring case and extra spaces
  status              same goal_status
  navigate_back, navigate_home, wait   type match is enough
`relaxed` also accepts the equivalences the AndroidControl paper allows:
  open_app      <-> clicking the element labelled with that app name
  navigate_back <-> clicking the on-screen back button
Only the action is scored; the thought is free text.

Not scored: steps without a usable reference (click outside the screen or on no element, scroll
whose direction could not be settled). They are counted under "unscorable".

    python score.py --data data/val.jsonl --pred runs/base/pred_val_high.jsonl --out runs/base/score_val_high.json
    python score.py --data data/val.jsonl --pred gold       # self-check: the reference answers as predictions
"""
import argparse
import collections
import json
import os
import random

import paths
import prompts as P

UNSCORABLE = {"oob", "no_element", "scroll_unclear"}
BACK_LABELS = {"navigate up", "back"}


resolve = paths.resolve


def norm(s):
    return " ".join((s or "").lower().split())


def inside(action, meta, box):
    px = P.action_to_px(action, meta["width"], meta["height"])
    l, t, r, b = box
    return l <= px["x"] < r and t <= px["y"] < b


def judge(pred, gold, meta):
    """-> (type_ok, strict_ok, relaxed_ok) for one step. pred may be None (unparseable)."""
    if pred is None:
        return False, False, False
    pt, gt = pred["action_type"], gold["action_type"]
    strict = False
    if pt == gt:
        if gt in ("click", "long_press"):
            strict = inside(pred, meta, meta["target_bbox"])
        elif gt == "scroll":
            strict = pred["direction"] == gold["direction"]
        elif gt == "input_text":
            strict = norm(pred["text"]) == norm(gold["text"])
        elif gt == "open_app":
            strict = norm(pred["app_name"]) == norm(gold["app_name"])
        elif gt == "status":
            strict = pred["goal_status"] == gold["goal_status"]
        else:
            strict = True
    relaxed = strict
    if not relaxed:
        if gt in ("open_app", "navigate_back") and pt == "click":
            relaxed = any(inside(pred, meta, box) for box in meta.get("equiv_click_bboxes", []))
        elif gt == "click" and pt == "open_app":
            relaxed = bool(meta.get("target_label")) and norm(pred["app_name"]) == norm(meta["target_label"])
        elif gt == "click" and pt == "navigate_back":
            relaxed = norm(meta.get("target_label")) in BACK_LABELS
    return pt == gt, strict, relaxed


def rate(items, key):
    return sum(i[key] for i in items) / len(items) if items else None


def bootstrap_ci(items, key, n_boot=1000, seed=0):
    """95% interval of the step rate, resampling whole episodes."""
    by_ep = collections.defaultdict(list)
    for i in items:
        by_ep[i["episode_id"]].append(i[key])
    eps = list(by_ep.values())
    if len(eps) < 2:
        return None
    rng, vals = random.Random(seed), []
    for _ in range(n_boot):
        num = den = 0
        for _ in range(len(eps)):
            e = eps[rng.randrange(len(eps))]
            num += sum(e)
            den += len(e)
        vals.append(num / den)
    vals.sort()
    return [round(vals[int(0.025 * n_boot)], 4), round(vals[int(0.975 * n_boot) - 1], 4)]


def summarize(items, ci=True):
    if not items:
        return {"n": 0}
    out = {"n": len(items)}
    for k in ("parsed", "type", "strict", "relaxed"):
        out[k] = round(rate(items, k), 4)
    if ci:
        out["strict_ci95"] = bootstrap_ci(items, "strict")
        out["relaxed_ci95"] = bootstrap_ci(items, "relaxed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True, help='predictions jsonl from predict.py, or "gold"')
    ap.add_argument("--out", help="write the full result as JSON")
    o = ap.parse_args()

    rows = [json.loads(line) for line in open(resolve(o.data), encoding="utf-8")]
    preds = None
    if o.pred != "gold":
        preds = {}
        for line in open(resolve(o.pred), encoding="utf-8"):
            d = json.loads(line)
            preds[(d["episode_id"], d["step"])] = d["pred_text"]

    items, unscorable, missing = [], collections.Counter(), 0
    for r in rows:
        m = r["meta"]
        key = (m["episode_id"], m["step"])
        if m["drop"] in UNSCORABLE:
            unscorable[m["drop"]] += 1
            continue
        gold = P.parse_output(r["messages"][2]["content"])["action"]
        if preds is None:
            text = r["messages"][2]["content"]
        elif key in preds:
            text = preds[key]
        else:
            missing += 1
            continue
        pred = P.parse_output(text)["action"]
        type_ok, strict, relaxed = judge(pred, gold, m)
        items.append({"episode_id": m["episode_id"], "step": m["step"], "gold_type": gold["action_type"],
                      "subsplits": m.get("subsplits", []), "parsed": pred is not None,
                      "type": type_ok, "strict": strict, "relaxed": relaxed})

    steps = [i for i in items if i["gold_type"] != "status"]
    result = {
        "data": o.data, "pred": o.pred,
        "rows": len(rows), "scored": len(items), "missing_prediction": missing, "unscorable": dict(unscorable),
        # the 8 AndroidControl actions only: the number comparable to AndroidControl results
        "androidcontrol_steps": summarize(steps),
        # including the terminal status step added per episode
        "all_steps": summarize(items),
        "by_action": {t: summarize([i for i in items if i["gold_type"] == t], ci=False)
                      for t in sorted({i["gold_type"] for i in items})},
    }
    subs = sorted({s for i in items for s in i["subsplits"]})
    if subs:
        result["by_subsplit"] = {s: summarize([i for i in steps if s in i["subsplits"]]) for s in subs}
    # episode success: every scored step of the episode correct (only episodes with all predictions)
    expected = collections.Counter(r["meta"]["episode_id"] for r in rows if r["meta"]["drop"] not in UNSCORABLE)
    got = collections.defaultdict(list)
    for i in items:
        got[i["episode_id"]].append(i["relaxed"])
    full = [all(v) for e, v in got.items() if len(v) == expected[e]]
    result["episodes"] = {"n": len(full), "all_steps_correct_relaxed": round(sum(full) / len(full), 4) if full else None}

    def line(name, s):
        if not s.get("n"):
            return f"{name:22s} n=0"
        ci = f"  95% {s['strict_ci95']}" if s.get("strict_ci95") else ""
        return (f"{name:22s} n={s['n']:5d}  parsed {s['parsed']:.3f}  type {s['type']:.3f}  "
                f"strict {s['strict']:.3f}{ci}  relaxed {s['relaxed']:.3f}")

    print(f"rows {len(rows)} | scored {len(items)} | missing prediction {missing} | unscorable {dict(unscorable)}")
    print(line("androidcontrol steps", result["androidcontrol_steps"]))
    print(line("all steps (+status)", result["all_steps"]))
    for t, s in result["by_action"].items():
        print(line("  " + t, s))
    for sname, s in result.get("by_subsplit", {}).items():
        print(line("  [" + sname + "]", s))
    print(f"episodes with every step correct: {result['episodes']}")
    if o.out:
        os.makedirs(os.path.dirname(resolve(o.out)), exist_ok=True)
        json.dump(result, open(resolve(o.out), "w"), indent=1)


if __name__ == "__main__":
    main()
