#!/usr/bin/env python3
"""Extract one AndroidControl TFRecord shard into plain files. Standard library only.

AndroidControl stores one episode per record (GZIP TFRecord of tf.train.Example):
  episode_id, goal, step_instructions, actions, screenshots (PNG bytes),
  accessibility_trees (serialized AndroidAccessibilityForest protos),
  screenshot_widths, screenshot_heights

Output (nothing is re-encoded or converted):
  <out>/images/episode_<id:06d>_step_<i:04d>.png   screenshot bytes as stored
  <out>/a11y/episode_<id:06d>_step_<i:04d>.pb      accessibility forest proto as stored
  <out>/meta/<shard>.jsonl                         one line per episode (goal, actions, ...)

Usage: extract_androidcontrol.py <shard_path> <out_dir>
"""
import gzip
import json
import os
import struct
import sys


def iter_records(path):
    """Yield raw record bytes. Raises on a truncated or corrupt file."""
    with gzip.open(path, "rb") as f:
        while True:
            head = f.read(8)
            if not head:
                return
            if len(head) < 8:
                raise IOError(f"truncated record header in {path}")
            (length,) = struct.unpack("<Q", head)
            f.read(4)
            data = f.read(length)
            if len(data) < length:
                raise IOError(f"truncated record body in {path}")
            f.read(4)
            yield data


def _varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) for a protobuf message."""
    pos, end = 0, len(buf)
    while pos < end:
        tag, pos = _varint(buf, pos)
        num, wire = tag >> 3, tag & 7
        if wire == 0:
            val, pos = _varint(buf, pos)
        elif wire == 1:
            val = buf[pos:pos + 8]
            pos += 8
        elif wire == 2:
            n, pos = _varint(buf, pos)
            val = buf[pos:pos + n]
            pos += n
        elif wire == 5:
            val = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield num, wire, val


def _signed(v):
    return v - (1 << 64) if v >= (1 << 63) else v


def _feature(buf):
    """tf.train.Feature -> list of bytes / int / float."""
    for num, _, val in _fields(buf):
        if num == 1:  # BytesList
            return [v for n, w, v in _fields(val) if n == 1 and w == 2]
        if num == 3:  # Int64List
            out = []
            for n, w, v in _fields(val):
                if n != 1:
                    continue
                if w == 2:  # packed
                    p = 0
                    while p < len(v):
                        x, p = _varint(v, p)
                        out.append(_signed(x))
                else:
                    out.append(_signed(v))
            return out
        if num == 2:  # FloatList
            out = []
            for n, w, v in _fields(val):
                if n != 1:
                    continue
                if w == 2:
                    out.extend(struct.unpack(f"<{len(v) // 4}f", v))
                else:
                    out.append(struct.unpack("<f", v)[0])
            return out
    return []


def parse_example(buf):
    """tf.train.Example -> {feature_name: list}."""
    feats = {}
    for num, _, features in _fields(buf):
        if num != 1:
            continue
        for n, _, entry in _fields(features):
            if n != 1:
                continue
            key, value = None, []
            for en, _, ev in _fields(entry):
                if en == 1:
                    key = bytes(ev).decode("utf-8")
                elif en == 2:
                    value = _feature(ev)
            feats[key] = value
    return feats


def write_if_needed(path, data):
    try:
        if os.path.getsize(path) == len(data):
            return
    except OSError:
        pass
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def main():
    shard, out = sys.argv[1], sys.argv[2]
    name = os.path.basename(shard)
    img_dir, a11y_dir, meta_dir = (os.path.join(out, d) for d in ("images", "a11y", "meta"))
    for d in (img_dir, a11y_dir, meta_dir):
        os.makedirs(d, exist_ok=True)

    n_ep = n_img = n_tree = img_bytes = tree_bytes = not_png = mismatch = 0
    meta_path = os.path.join(meta_dir, name + ".jsonl")
    with open(meta_path + ".tmp", "w", encoding="utf-8") as meta:
        for rec in iter_records(shard):
            ex = parse_example(memoryview(rec))
            ep = int(ex["episode_id"][0])
            shots = ex.get("screenshots", [])
            trees = ex.get("accessibility_trees", [])
            actions = [json.loads(bytes(a).decode("utf-8")) for a in ex.get("actions", [])]
            images, a11y = [], []
            for i, b in enumerate(shots):
                if bytes(b[:4]) != b"\x89PNG":
                    not_png += 1
                fn = f"episode_{ep:06d}_step_{i:04d}.png"
                write_if_needed(os.path.join(img_dir, fn), b)
                images.append("images/" + fn)
                img_bytes += len(b)
            for i, b in enumerate(trees):
                fn = f"episode_{ep:06d}_step_{i:04d}.pb"
                write_if_needed(os.path.join(a11y_dir, fn), b)
                a11y.append("a11y/" + fn)
                tree_bytes += len(b)
            if len(shots) != len(trees) or len(shots) != len(actions) + 1:
                mismatch += 1
            meta.write(json.dumps({
                "episode_id": ep,
                "goal": bytes(ex["goal"][0]).decode("utf-8"),
                "step_instructions": [bytes(s).decode("utf-8") for s in ex.get("step_instructions", [])],
                "actions": actions,
                "screenshot_widths": list(ex.get("screenshot_widths", [])),
                "screenshot_heights": list(ex.get("screenshot_heights", [])),
                "images": images,
                "accessibility_trees": a11y,
                "shard": name,
            }, ensure_ascii=False) + "\n")
            n_ep += 1
            n_img += len(shots)
            n_tree += len(trees)
    os.replace(meta_path + ".tmp", meta_path)
    print(json.dumps({"shard": name, "episodes": n_ep, "images": n_img, "trees": n_tree,
                      "image_bytes": img_bytes, "tree_bytes": tree_bytes,
                      "not_png": not_png, "count_mismatch_episodes": mismatch}), flush=True)


if __name__ == "__main__":
    main()
