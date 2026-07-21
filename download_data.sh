#!/bin/bash
#  XmoPipe - Dataset download script
#
#  Downloads the 263D dataset (Data_NPY folder on the project drive) and
#  extracts it into ./datasets/. Model checkpoints are in download.sh.
#
#  Usage:  ./download_data.sh [drive-folder-url] [destination]
#          Both default below; pass a URL only if the drive folder moved.
#  Needs:  gdown, unzip
#
#    datasets/
#    ├─ new_joint_vecs/*.npy   # 263D vectors, ~61 GB
#    ├─ texts/*.txt            # captions
#    ├─ metadatas/             # per-video metadata
#    ├─ Mean.npy  Std.npy      # normalization stats
#    └─ train.txt  val.txt  test.txt  train_val.txt  all.txt
#
#  Expect ~120 GB free while it runs: each archive is deleted as soon as it is
#  extracted, but the largest one briefly coexists with its contents.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DRIVE="https://drive.google.com/drive/folders/1Tp9QsLZAoFEckvSTPw1VTI6ynk2UgFit"
DATA_NPY="https://drive.google.com/drive/folders/1SjOpbF8QIIe628Llc-P57dFaprOBoRGn"

URL="${1:-${XMOPIPE_DATA_NPY_URL:-$DATA_NPY}}"
DEST="${2:-$REPO_DIR/datasets}"

# What the training loader needs; verified after extraction to catch a partial
# download. metadatas is installed too but training never reads it.
EXPECTED="new_joint_vecs texts Mean.npy Std.npy train.txt val.txt test.txt all.txt"

fail() { echo "  !!! $*" >&2; exit 1; }

# The gdown console script breaks when the env it was installed into is gone,
# so prefer the module.
if   python3 -c "import gdown" 2>/dev/null; then GDOWN="python3 -m gdown"
elif command -v gdown >/dev/null;           then GDOWN=gdown
else fail "gdown not found: pip install gdown"; fi
command -v unzip >/dev/null || fail "unzip not found: sudo apt install unzip"

echo "  Destination: $DEST"
mkdir -p "$DEST"

# Already there? Do not spend 61 GB finding out. A partial install redownloads
# everything, but gdown --continue resumes files left behind mid-download.
MISSING=""
for item in $EXPECTED; do [ -e "$DEST/$item" ] || MISSING="$MISSING $item"; done
[ -z "$MISSING" ] && { echo "  ==> already installed, nothing to do."; exit 0; }
[ "$(ls -A "$DEST" 2>/dev/null)" ] && echo "  Missing:$MISSING"

FREE=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc 0-9)
if [ "${FREE:-0}" -lt 120 ]; then
  echo "  !!! Only ${FREE} GB free, ~120 GB recommended."
  [ -t 0 ] || fail "Stopping (unattended run)."
  read -n 1 -r -p "      Continue anyway? [y/N]: " k; echo
  [[ $k =~ [yY] ]] || exit 1
fi

# Stage inside DEST so installing is a rename, not a 61 GB copy.
STAGING="$DEST/.download_staging"
mkdir -p "$STAGING"

echo "  Downloading Data_NPY (this takes a while)..."
$GDOWN --folder "$URL" -O "$STAGING" --continue --remaining-ok || true

# gdown nests the contents in a folder named after the drive folder. It also
# exits 0 when it could not read the folder at all, so trust the payload rather
# than the status: Mean.npy is the small file that always ships with Data_NPY.
SRC="$STAGING"
[ -e "$SRC/Mean.npy" ] || SRC=$(find "$STAGING" -mindepth 1 -maxdepth 1 -type d | head -1)
[ -n "$SRC" ] && [ -e "$SRC/Mean.npy" ] || fail "Download failed, nothing usable came back.
      If the folder moved or is no longer shared, get a fresh link:
      open $DRIVE, right-click Data_NPY -> Share -> Copy link, then:
      ./download_data.sh '<link>'"

# The payload folders are zipped on the drive (gdown caps folder downloads at 50
# files). Archives may sit at the top or inside a folder of the same name; they
# store their own top-level directory, so always extract into SRC.
while IFS= read -r z; do
  echo "  Extracting $(basename "$z")"
  unzip -q -o "$z" -d "$SRC" || fail "$(basename "$z") is corrupt. Delete $STAGING and retry."
  rm -f "$z"   # freed now, the extracted copy is what we keep
done < <(find "$SRC" -name '*.zip')

# The drive spells it new_joints_vecs, the loader wants new_joint_vecs.
[ -d "$SRC/new_joints_vecs" ] && mv -T "$SRC/new_joints_vecs" "$SRC/new_joint_vecs"

# Undo the extra level when an archive shipped inside a folder of the same name.
for item in new_joint_vecs texts metadatas; do
  [ -d "$SRC/$item/$item" ] && mv "$SRC/$item/$item" "$SRC/.f" && rm -rf "$SRC/$item" && mv "$SRC/.f" "$SRC/$item"
done

for item in $EXPECTED metadatas train_val.txt; do
  [ -e "$SRC/$item" ] && rm -rf "${DEST:?}/$item" && mv "$SRC/$item" "$DEST/"
done
rm -rf "$STAGING"

for item in $EXPECTED; do
  [ -e "$DEST/$item" ] || fail "Install incomplete, missing: $item"
done

# A folder that was not zipped on the drive comes back truncated to gdown's
# 50-file cap, which otherwise looks like a perfectly valid install.
N=$(ls "$DEST/new_joint_vecs" | wc -l)
[ "$N" -le 50 ] && fail "Only $N motions in new_joint_vecs: gdown caps folder
      downloads at 50 files, so this folder was not zipped on the drive.
      Zip it there, or fetch it with rclone instead."

echo "  ==> installed in $DEST"
echo "      $N motions, $(du -sh "$DEST" | cut -f1) total"
