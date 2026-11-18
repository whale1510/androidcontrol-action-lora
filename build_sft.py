#!/usr/bin/env python3
"""AndroidControl -> SFT jsonl for the action LoRA.

One example per step: (goal, history, current screenshot) -> thought + next action.
The thought is AndroidControl's step instruction. Each episode also gets one terminal example:
last screenshot -> status action.

    python build_sft.py                      # reads data/androidcontrol (or $ANDROIDCONTROL_DIR)

Output (data/):
    train.jsonl  official train split; steps excluded by a rule below are not used as targets
    val.jsonl    official validation split, every step (meta.drop names the rule it fails)
    test.jsonl   official test split, every step (meta.subsplits lists IDD / *_unseen)
    scroll_fixes.jsonl  every scroll whose direction was corrected or excluded, with the evidence
    report.json  counts per split, per action type, per rule
    dataset_info.json  so the same files load in LLaMA-Factory (formatting: sharegpt)

Scroll direction check. Three sources are compared for every scroll:
    L  the recorded label
    O  the direction implied by how labelled elements moved between the accessibility trees
       before and after (None when fewer than 3 elements moved)
    P  the swipe/scroll phrase in the step instruction ("swipe up", "scroll left", ...)
  The axis of P (vertical / horizontal) is reliable. Its direction is reliable only for
  "swipe up" (-> down), "swipe left" and "scroll left" (-> right): >= 99% on the steps where
  L and O agree. Decisions:
    O == L                                   keep
    O missing, P on the other axis than L    fix to P's direction if reliable, else exclude
    O missing, otherwise                     keep
    O on the other axis than L               P on O's axis -> fix to O; P on L's axis -> keep
                                             (the measurement is the outlier); no P -> exclude
    O opposite to L on the same axis         P's reliable direction == L -> keep; == O -> fix to O;
                                             otherwise exclude
  A fixed direction replaces the label everywhere (target and later history). The original
  action is kept in meta.source_action.

Steps never used as training targets (they stay in later steps' history):
    oob             click/long_press outside the screenshot
    no_element      click/long_press that hits no visible accessibility node
    scroll_unclear  scroll excluded by the check above
    no_instruction  empty step instruction (nothing to write as the thought)
"""
import argparse
import collections
import json
import os
import re
import statistics
from multiprocessing import Pool

import a11y_proto
import paths
import prompts as P

HERE = paths.HERE
AC = paths.androidcontrol_dir()
MOVE_MIN_PX = 20       # a node counts as moved if it shifted more than this
MOVE_MIN_NODES = 3     # need at least this many moved nodes to trust the measurement
AXIS = {"up": "v", "down": "v", "left": "h", "right": "h"}
RELIABLE_PHRASE = {("swipe", "up"): "down", ("swipe", "left"): "right", ("scroll", "left"): "right"}
PHRASE = re.compile(
    r"\b(swipe|scroll|slide|drag)\w*\s+(?:the\s+\w+\s+)?"
    r"(?:from\s+\w+\s+to\s+(?:the\s+)?|to\s+(?:the\s+)?|towards?\s+(?:the\s+)?)?"
    r"(up|down|left|right)(?:ward|wards)?\b")
BACK_LABELS = {"navigate up", "back"}


def norm(s):
    return " ".join((s or "").lower().split())


def load_forest(rel_path):
    return a11y_proto.load_forest(os.path.join(AC, rel_path))


def visible_nodes(forest):
    for w in forest.windows:
        for n in w.tree.nodes:
            b = n.bounds_in_screen
            if n.is_visible_to_user and b.right > b.left and b.bottom > b.top:
                yield n, [b.left, b.top, b.right, b.bottom]


def smallest_node(forest, x, y):
    """(bbox, label) of the smallest visible node containing (x, y), or (None, None)."""
    best = None
    for n, (l, t, r, b) in visible_nodes(forest):
        if l <= x < r and t <= y < b:
            area = (r - l) * (b - t)
            if best is None or area < best[0]:
                best = (area, [l, t, r, b], (n.text or n.content_description).strip())
    return (best[1], best[2]) if best else (None, None)


def boxes_labelled(forest, labels):
    """Bounding boxes of visible nodes whose text or description is one of `labels` (normalised)."""
    out = []
    for n, box in visible_nodes(forest):
        if norm(n.text) in labels or norm(n.content_description) in labels:
            out.append(box)
    return out


def text_nodes(forest):
    """{(text, class): (cx, cy)} for labelled visible nodes whose key is unique on the screen."""
    seen = collections.defaultdict(list)
    for w in forest.windows:
        for n in w.tree.nodes:
            t = (n.text or n.content_description).strip()
            if t and n.is_visible_to_user:
                b = n.bounds_in_screen
                seen[(t, n.class_name)].append(((b.left + b.right) / 2, (b.top + b.bottom) / 2))
    return {k: v[0] for k, v in seen.items() if len(v) == 1}


def observed_scroll(before, after):
    """Scroll direction implied by how the content moved, or None if it cannot be measured.

    Content moving up means content below was revealed = "down" (same meaning as AndroidWorld).
    """
    a, b = text_nodes(before), text_nodes(after)
    d = [(b[k][0] - a[k][0], b[k][1] - a[k][1]) for k in a if k in b]
    d = [v for v in d if abs(v[0]) > MOVE_MIN_PX or abs(v[1]) > MOVE_MIN_PX]
    if len(d) < MOVE_MIN_NODES:
        return None
    dx, dy = statistics.median(v[0] for v in d), statistics.median(v[1] for v in d)
    if abs(dy) >= abs(dx):
        return "down" if dy < 0 else "up"
    return "right" if dx < 0 else "left"


def scroll_phrase(instruction):
    m = PHRASE.search(instruction.lower())
    return (m.group(1), m.group(2)) if m else None


def decide_scroll(label, observed, phrase):
    """-> (decision, direction). decision is 'keep', 'fix' or 'exclude'. See the module docstring."""
    reliable = RELIABLE_PHRASE.get(phrase)
    p_axis = AXIS[phrase[1]] if phrase else None
    if observed == label:
        return "keep", label
    if observed is None:
        if phrase is not None and p_axis != AXIS[label]:
            return ("fix", reliable) if reliable else ("exclude", label)
        return "keep", label
    if AXIS[observed] != AXIS[label]:
        if phrase is None:
            return "exclude", label
        return ("fix", observed) if p_axis == AXIS[observed] else ("keep", label)
    if reliable == label:
        return "keep", label
    if reliable == observed:
        return "fix", observed
    return "exclude", label


def process_episode(job):
    ep, split, subsplits = job
    n = len(ep["actions"])
    W, H = ep["screenshot_widths"], ep["screenshot_heights"]
    forests = {}

    def forest(i):
        if i not in forests:
            forests[i] = load_forest(ep["accessibility_trees"][i])
        return forests[i]

    # pass 1: decide the final action of every step
    steps, fixes = [], []
    for i, src in enumerate(ep["actions"]):
        t = src["action_type"]
        action, drop, extra = dict(src), None, {}
        thought = P.clean_thought(ep["step_instructions"][i])
        if t in ("click", "long_press"):
            if not (0 <= src["x"] < W[i] and 0 <= src["y"] < H[i]):
                drop = "oob"
            else:
                bbox, label = smallest_node(forest(i), src["x"], src["y"])
                if bbox is None:
                    drop = "no_element"
                else:
                    extra["target_bbox"] = bbox          # pixels; a click inside it counts as correct
                    if label:
                        extra["target_label"] = label
        elif t == "scroll":
            observed = observed_scroll(forest(i), forest(i + 1))
            phrase = scroll_phrase(ep["step_instructions"][i])
            decision, direction = decide_scroll(src["direction"], observed, phrase)
            if decision != "keep":
                fixes.append({"episode_id": ep["episode_id"], "step": i, "split": split, "decision": decision,
                              "label": src["direction"], "observed": observed,
                              "phrase": " ".join(phrase) if phrase else None,
                              "final": direction if decision == "fix" else None,
                              "instruction": ep["step_instructions"][i]})
            if decision == "fix":
                action["direction"] = direction
                extra["scroll_fixed_from"] = src["direction"]
            elif decision == "exclude":
                drop = "scroll_unclear"
        elif t == "open_app":
            boxes = boxes_labelled(forest(i), {norm(src["app_name"])})
            if boxes:
                extra["equiv_click_bboxes"] = boxes       # tapping the app's icon is the same action
        elif t == "navigate_back":
            boxes = boxes_labelled(forest(i), BACK_LABELS)
            if boxes:
                extra["equiv_click_bboxes"] = boxes       # tapping the on-screen back button
        if not thought and drop is None:
            drop = "no_instruction"
        steps.append({"source": src, "rel": P.action_to_rel(action, W[i], H[i]), "thought": thought,
                      "drop": drop, "extra": extra})
    steps.append({"source": dict(P.STATUS_ACTION), "rel": dict(P.STATUS_ACTION), "thought": P.STATUS_THOUGHT,
                  "drop": None, "extra": {}})

    # pass 2: one example per step, history = the final thought and action of the previous steps
    rows = []
    for i, s in enumerate(steps):
        history = [{"thought": p["thought"], "action": p["rel"]} for p in steps[:i]]
        meta = {"episode_id": ep["episode_id"], "step": i, "num_steps": n, "split": split,
                "width": W[i], "height": H[i],
                "source_action": s["source"],            # AndroidControl action as stored (pixels)
                "drop": s["drop"], **s["extra"]}
        if subsplits:
            meta["subsplits"] = subsplits
        rows.append({"messages": P.build_messages(ep["goal"], history, s["thought"], s["rel"]),
                     "images": [ep["images"][i]], "meta": meta})
    return rows, fixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes per split (for a quick check)")
    o = ap.parse_args()
    os.makedirs(o.out, exist_ok=True)

    splits = json.load(open(os.path.join(AC, "raw", "splits.json")))
    sub = json.load(open(os.path.join(AC, "raw", "test_subsplits.json")))
    which = {e: k for k, ids in splits.items() for e in ids}
    subs = collections.defaultdict(list)
    for k, ids in sub.items():
        for e in ids:
            subs[e].append(k)

    jobs = collections.defaultdict(list)
    with open(os.path.join(AC, "episodes.jsonl")) as f:
        for line in f:
            ep = json.loads(line)
            s = which[ep["episode_id"]]
            if not o.limit or len(jobs[s]) < o.limit:
                jobs[s].append((ep, s, sorted(subs.get(ep["episode_id"], []))))

    report = {"max_history": P.MAX_HISTORY, "splits": {}}
    names = {"train": "train", "validation": "val", "test": "test"}
    fix_path = os.path.join(o.out, "scroll_fixes.jsonl")
    with Pool(o.workers) as pool, open(fix_path + ".tmp", "w", encoding="utf-8") as fix_out:
        for split, fname in names.items():
            c = collections.Counter()
            by_type, dropped, fixed = collections.Counter(), collections.Counter(), collections.Counter()
            path = os.path.join(o.out, fname + ".jsonl")
            with open(path + ".tmp", "w", encoding="utf-8") as out:
                for rows, fixes in pool.imap(process_episode, jobs[split], chunksize=8):
                    c["episodes"] += 1
                    for fx in fixes:
                        fix_out.write(json.dumps(fx, ensure_ascii=False) + "\n")
                        if fx["decision"] == "fix":
                            fixed[f"{fx['label']}->{fx['final']}"] += 1
                    for r in rows:
                        c["steps_total"] += 1
                        d = r["meta"]["drop"]
                        if d:
                            dropped[d] += 1
                            if split == "train":
                                continue                     # never a training target
                        by_type[r["meta"]["source_action"]["action_type"]] += 1
                        c["written"] += 1
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(path + ".tmp", path)
            report["splits"][fname] = {**c, "excluded_as_target": dict(dropped), "scroll_fixed": dict(fixed),
                                       "written_by_action": dict(by_type.most_common())}
            print(fname, json.dumps(report["splits"][fname]), flush=True)
    os.replace(fix_path + ".tmp", fix_path)

    json.dump(report, open(os.path.join(o.out, "report.json"), "w"), indent=1)
    tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
            "system_tag": "system"}
    info = {f"androidcontrol_{n}": {"file_name": f"{n}.jsonl", "formatting": "sharegpt",
                                     "columns": {"messages": "messages", "images": "images"}, "tags": tags}
            for n in names.values()}
    json.dump(info, open(os.path.join(o.out, "dataset_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
