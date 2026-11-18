#!/usr/bin/env python3
"""Verify data/*.jsonl against the original AndroidControl records. Exits non-zero on any failure.

    python check_data.py
"""
import collections
import json
import os
import sys

import paths
import prompts as P

HERE = paths.HERE
AC = paths.androidcontrol_dir()


def main():
    episodes = {}
    with open(os.path.join(AC, "episodes.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            episodes[e["episode_id"]] = e
    images = set(os.listdir(os.path.join(AC, "images")))
    fixed = {}                                   # (episode, step) -> corrected scroll direction
    for line in open(os.path.join(HERE, "data", "scroll_fixes.jsonl")):
        d = json.loads(line)
        if d["decision"] == "fix":
            fixed[(d["episode_id"], d["step"])] = d["final"]

    def final_step(ep, j):
        """(thought, model-space action) of step j as the dataset should contain it."""
        if j == len(ep["actions"]):
            return P.STATUS_THOUGHT, dict(P.STATUS_ACTION)
        a = dict(ep["actions"][j])
        if (ep["episode_id"], j) in fixed:
            a["direction"] = fixed[(ep["episode_id"], j)]
        return (P.clean_thought(ep["step_instructions"][j]),
                P.action_to_rel(a, ep["screenshot_widths"][j], ep["screenshot_heights"][j]))

    bad = collections.Counter()
    for name in ("train", "val", "test"):
        n = clicks = off_element = n_fixed = 0
        max_px_err = 0
        seen = set()
        for line in open(os.path.join(HERE, "data", name + ".jsonl")):
            r = json.loads(line)
            m, n = r["meta"], n + 1
            ep, i = episodes[m["episode_id"]], m["step"]
            sys_msg, user, ans = r["messages"]
            key = (m["episode_id"], i)
            bad["duplicate"] += key in seen
            seen.add(key)
            bad["system"] += sys_msg["content"] != P.SYSTEM_PROMPT
            bad["image_tag"] += user["content"].count(P.IMAGE_TAG) != 1
            bad["image_missing"] += os.path.basename(r["images"][0]) not in images
            bad["image_path"] += r["images"][0] != ep["images"][i]
            src = dict(P.STATUS_ACTION) if i == len(ep["actions"]) else ep["actions"][i]
            bad["source_mismatch"] += src != m["source_action"]
            # answer: thought = the step instruction, action = original action in 0..999 (scroll fix applied)
            thought, want = final_step(ep, i)
            out = P.parse_output(ans["content"])
            bad["unparseable"] += out["action"] is None
            bad["target_mismatch"] += out["action"] != want
            bad["thought_mismatch"] += (out["thought"] or "") != thought
            bad["text_not_canonical"] += ans["content"] != P.answer_text(thought, want)
            bad["fix_flag"] += (key in fixed) != ("scroll_fixed_from" in m)
            n_fixed += key in fixed
            # user turn: goal + thought and action of every previous step
            hist = [dict(zip(("thought", "action"), final_step(ep, j))) for j in range(i)]
            bad["user_text"] += user["content"] != P.user_text(ep["goal"], hist)
            act = out["action"]
            if act and "x" in act:
                clicks += 1
                px = P.action_to_px(act, m["width"], m["height"])
                if not m["drop"]:
                    max_px_err = max(max_px_err, abs(px["x"] - src["x"]), abs(px["y"] - src["y"]))
                    l, t, rr, b = m["target_bbox"]
                    # 0..999 is coarser than the screen (about 1 px in x, 2.4 px in y on 1080x2400),
                    # so a click on a very thin element can leave it after rounding. Reported, not a failure.
                    off_element += not (l <= px["x"] < rr and t <= px["y"] < b)
            if name == "train":
                bad["train_has_excluded"] += bool(m["drop"])
        print(f"{name}: rows {n}, scroll fixes applied {n_fixed}, click/long_press {clicks}, "
              f"max pixel error after 0-999 round trip {max_px_err}, "
              f"clicks leaving their element after rounding {off_element}")
    fails = {k: v for k, v in bad.items() if v}
    print("failures:", fails or "none")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
