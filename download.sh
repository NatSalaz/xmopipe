#!/bin/bash
#  XmoPipe - Model download script
#
#  Run once after setup.sh to download all required checkpoints.
#  Requirements: gdown, wget  (both available in xmo-3d env)
#
#  Models requiring manual registration are flagged below.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

urle() { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

dl_file() {
  local path="$1"; shift
  if [ -f "$path" ]; then
    echo "  ==> $(basename "$path") already exists, skipping."
  else
    "$@"
  fi
}

echo " XmoPipe - Model Downloads"

# 1 - GVHMR checkpoints
echo ""
echo "1/6 - GVHMR checkpoints"
mkdir -p "$REPO_DIR/3-Body/GVHMR/inputs/checkpoints"
CKPT_DIR="$REPO_DIR/3-Body/GVHMR/inputs/checkpoints"

mkdir -p "$CKPT_DIR/dpvo" "$CKPT_DIR/gvhmr" "$CKPT_DIR/hmr2" "$CKPT_DIR/vitpose" "$CKPT_DIR/yolo"
dl_file "$CKPT_DIR/dpvo/dpvo.pth" \
  gdown "1DE5GVftRCfZOTMp8YWF0xkGudDxK0nr0" -O "$CKPT_DIR/dpvo/dpvo.pth"
dl_file "$CKPT_DIR/gvhmr/gvhmr_siga24_release.ckpt" \
  gdown "1c9iCeKFN4Kr6cMPJ9Ss6Jdc3SZFnO5NP" -O "$CKPT_DIR/gvhmr/gvhmr_siga24_release.ckpt"
dl_file "$CKPT_DIR/hmr2/epoch=10-step=25000.ckpt" \
  gdown "1X5hvVqvqI9tvjUCb2oAlZxtgIKD9kvsc" -O "$CKPT_DIR/hmr2/epoch=10-step=25000.ckpt"
dl_file "$CKPT_DIR/vitpose/vitpose-h-multi-coco.pth" \
  gdown "1sR8xZD9wrZczdDVo6zKscNLwvarIRhP5" -O "$CKPT_DIR/vitpose/vitpose-h-multi-coco.pth"
dl_file "$CKPT_DIR/yolo/yolov8x.pt" \
  gdown "1_HGm-lqIH83-M1ML4bAXaqhm_eT2FKo5" -O "$CKPT_DIR/yolo/yolov8x.pt"

# 2 - SMPL body models  (requires free registration at https://smpl.is.tue.mpg.de/)
echo ""
echo "2/6 - SMPL body models - registration required"
if [ -f "$CKPT_DIR/body_models/smpl/SMPL_NEUTRAL.pkl" ]; then
  echo "  ==> Already in place, skipping."
else
  echo "      Sign up at https://smpl.is.tue.mpg.de/ then place files at:"
  echo "      3-Body/GVHMR/inputs/checkpoints/body_models/smpl/"
  read -p "      Press Enter once files are in place (or Ctrl+C to skip)"
fi

# 3 - SMPLX body models  (requires free registration at https://smpl-x.is.tue.mpg.de/)
echo ""
echo "3/6 - SMPLX body models - registration required"
if [ -f "$CKPT_DIR/body_models/smplx/SMPLX_NEUTRAL.pkl" ]; then
  echo "  ==> Already in place, skipping."
else
  echo "      Sign up at https://smpl-x.is.tue.mpg.de/ then place files at:"
  echo "      3-Body/GVHMR/inputs/checkpoints/body_models/smplx/"
  read -p "      Press Enter once files are in place (or Ctrl+C to skip)"
fi

# 4 - SMIRK checkpoints
echo ""
echo "4/6 - SMIRK checkpoints"
cd "$REPO_DIR/4-Face/smirk"
mkdir -p trained_models assets

dl_file "trained_models/SMIRK_em1.pt" \
  gdown "https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view" -O trained_models/SMIRK_em1.pt --fuzzy

dl_file "assets/face_landmarker.task" \
  wget -q --show-progress \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
    -O assets/face_landmarker.task

if [ -f "assets/FLAME2020/generic_model.pkl" ]; then
  echo "  ==> FLAME2020 already in place, skipping."
else
  echo "  ==> FLAME2020 (registration required at https://flame.is.tue.mpg.de/)"
  read -p "     Username (FLAME): " username
  read -sp "    Password (FLAME): " password
  echo ""
  username=$(urle "$username")
  password=$(urle "$password")
  wget --post-data "username=$username&password=$password" \
    'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' \
    -O FLAME2020.zip --no-check-certificate --continue
  mkdir -p assets/FLAME2020
  unzip -q FLAME2020.zip -d assets/FLAME2020/
  rm FLAME2020.zip
fi

dl_file "yolov8l_100e.pt" \
  gdown "https://drive.google.com/file/d/1iHL-XjvzpbrE8ycVqEbGla4yc1dWlSWU/view" -O yolov8l_100e.pt --fuzzy

dl_file "fer2013_model.pth" \
  gdown --folder "https://drive.google.com/drive/folders/1Tp9QsLZAoFEckvSTPw1VTI6ynk2UgFit" -O "$REPO_DIR/4-Face/smirk" --remaining-ok

# 5 - FastSAM-x.pt
echo ""
echo "5/6 - FastSAM-x.pt"
dl_file "$REPO_DIR/6-Captions/FastSAM-x.pt" \
  wget -q --show-progress \
    "https://huggingface.co/CASIA-LMC-Lab/FastSAM/resolve/main/FastSAM-x.pt" \
    -O "$REPO_DIR/6-Captions/FastSAM-x.pt"

# 6 - yolo11n-pose.pt  (auto-downloaded by ultralytics on first run)
echo ""
echo "6/6 - yolo11n-pose.pt - auto-downloaded by Ultralytics on first filter run."

echo ""
echo " Download complete."
echo " Only manual steps remaining: SMPL and SMPLX body models (registration required)."