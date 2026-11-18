#!/bin/bash
# Download AndroidControl from the public bucket, verify it, and extract it into plain files.
#
#   bash scripts/get_androidcontrol.sh                 # into data/androidcontrol
#   ANDROIDCONTROL_DIR=/big/disk/androidcontrol bash scripts/get_androidcontrol.sh
#   SHARDS="00000" bash scripts/get_androidcontrol.sh  # quick trial with one of the 20 shards
#
# Result (about 100 GB for all 20 shards):
#   raw/       20 GZIP TFRecord shards + splits.json + test_subsplits.json   (47 GB)
#   images/    episode_<id>_step_<i>.png, original resolution                  (48 GB, 99,131 files)
#   a11y/      episode_<id>_step_<i>.pb, accessibility trees as stored         (4.9 GB, 99,131 files)
#   episodes.jsonl   one line per episode: goal, step instructions, actions, file paths
# Re-runnable: finished downloads are skipped, extraction skips files already written.
# Needs: wget, md5sum, python3 (standard library only).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ANDROIDCONTROL_DIR:-$REPO/data/androidcontrol}"
PYTHON="${PYTHON:-python3}"
DOWNLOADS="${DOWNLOADS:-4}"     # parallel downloads
WORKERS="${WORKERS:-8}"         # parallel extraction processes
ALL_SHARDS="$(seq -w 0 19 | sed 's/^/000/')"
SHARDS="${SHARDS:-$ALL_SHARDS}"
BASE=https://storage.googleapis.com/gresearch/android_control
MD5="$REPO/scripts/androidcontrol_md5.txt"      # "<md5> <bytes> <name>" from the bucket listing
RAW="$DEST/raw"; LOG="$DEST/logs"
mkdir -p "$RAW" "$LOG"

FILES="splits.json test_subsplits.json"
for s in $SHARDS; do FILES="$FILES android_control-$s-of-00020"; done

need() {   # file is missing or not the expected size
  local want; want=$(awk -v n="$1" '$3==n{print $2}' "$MD5")
  [ ! -f "$RAW/$1" ] || [ "$(stat -L -c %s "$RAW/$1")" != "$want" ]
}

echo "[$(date +%T)] destination: $DEST"
TODO=""; for f in $FILES; do need "$f" && TODO="$TODO $f"; done
if [ -n "$TODO" ]; then
  echo "[$(date +%T)] downloading:$TODO"
  printf '%s\n' $TODO | xargs -P "$DOWNLOADS" -I{} wget -c -q --tries=5 -O "$RAW/{}" "$BASE/{}"
fi
for f in $FILES; do
  if need "$f"; then echo "ERROR: $f is missing or incomplete. Run the script again."; exit 1; fi
done

echo "[$(date +%T)] verifying md5"
: > "$LOG/md5.log"
for f in $FILES; do
  echo "$(awk -v n="$f" '$3==n{print $1}' "$MD5")  $RAW/$f"
done | xargs -P 8 -L 1 bash -c 'echo "$0  $1" | md5sum -c --quiet - >> "'"$LOG/md5.log"'" 2>&1'
if [ -s "$LOG/md5.log" ]; then echo "ERROR: md5 mismatch:"; cat "$LOG/md5.log"; exit 1; fi

echo "[$(date +%T)] extracting"
for s in $SHARDS; do echo "$RAW/android_control-$s-of-00020"; done \
  | xargs -P "$WORKERS" -I{} "$PYTHON" "$REPO/scripts/extract_androidcontrol.py" {} "$DEST" \
  > "$LOG/extract.log" 2> "$LOG/extract.err"
if [ $? -ne 0 ] || [ -s "$LOG/extract.err" ]; then echo "ERROR: extraction failed, see $LOG/extract.err"; exit 1; fi

# episodes.jsonl in shard order (the order decides the order of the training file)
: > "$DEST/episodes.jsonl.tmp"
for s in $SHARDS; do cat "$DEST/meta/android_control-$s-of-00020.jsonl" >> "$DEST/episodes.jsonl.tmp"; done
mv "$DEST/episodes.jsonl.tmp" "$DEST/episodes.jsonl"
echo "[$(date +%T)] done: $(wc -l < "$DEST/episodes.jsonl") episodes, $(ls -f "$DEST/images" | grep -c '\.png$') screenshots, $(ls -f "$DEST/a11y" | grep -c '\.pb$') trees"
echo "expected for the full dataset: 15283 episodes, 99131 screenshots, 99131 trees"
