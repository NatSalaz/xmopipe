#!/bin/bash
#  XmoPipe - Model download script
#
#  Run once after setup.sh to download all required checkpoints.
#  Requirements: gdown, wget
#
#  Each step asks before running: Enter runs it, s skips it.
#  Run everything unattended with:  ./download.sh -y   (or XMOPIPE_YES=1)
#
#  Models requiring manual registration are flagged below.
#
#  Steps 2 and 3 must leave this tree behind:
#
#    3-Body/GVHMR/inputs/checkpoints/
#    ├─ body_models/smplx/
#    |  └─ SMPLX_{GENDER}.npz   # SMPLX (predicted params + evaluation)
#    └─ body_models/smpl/
#       └─ SMPL_{GENDER}.pkl    # SMPL  (rendering and evaluation)
#
#  {GENDER} is NEUTRAL, MALE and FEMALE. NEUTRAL alone is enough to run the
#  pipeline; the other two are only needed for gendered evaluation.

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run every step without asking: ./download.sh -y, or XMOPIPE_YES=1 ./download.sh
AUTO_YES="${XMOPIPE_YES:-0}"
case "${1:-}" in
  -y|--yes) AUTO_YES=1 ;;
esac

# Ask before running a step. Enter or y runs it, s skips it.
# Runs the step unattended when --yes was passed or stdin is not a terminal.
ask_step() {
  local prompt="$1" key
  if [ "$AUTO_YES" = "1" ] || [ ! -t 0 ]; then
    return 0
  fi
  while true; do
    if ! read -n 1 -r -p "      $prompt  [Enter] run / [s] skip: " key; then
      echo ""
      return 0
    fi
    echo ""
    case "$key" in
      ""|y|Y) return 0 ;;
      s|S)    echo "      ==> skipped."; return 1 ;;
      *)      echo "      '$key'? Press Enter to run, or s to skip." ;;
    esac
  done
}

# Same prompt, for steps that cannot work without a human (manual file drops,
# credential entry). Unattended runs skip these instead of running them blind.
ask_manual_step() {
  if [ "$AUTO_YES" = "1" ] || [ ! -t 0 ]; then
    echo "      ==> needs input, skipped (unattended run)."
    return 1
  fi
  ask_step "$1"
}

urle() { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL, SMPLX and FLAME all live behind the same MPI account on
# download.is.tue.mpg.de, so ask for the login once and reuse it.
# Each license still has to be accepted once on its own website.
MPI_USER=""
MPI_PASS=""
mpi_credentials() {
  [ -n "$MPI_USER" ] && return 0
  echo "      One login covers SMPL, SMPLX and FLAME (same MPI account)."
  read -p  "      Username (email): " MPI_USER
  read -sp "      Password: " MPI_PASS
  echo ""
  MPI_USER=$(urle "$MPI_USER")
  MPI_PASS=$(urle "$MPI_PASS")
}

# Fetch a registration-gated archive from download.is.tue.mpg.de.
# Neither failure mode looks like a failure to the caller: a bad login is a 401
# that leaves a 0-byte file, and an unaccepted license is a 200 carrying an HTML
# page saved under the .zip name. Verify the payload is really an archive, or
# the breakage only surfaces much later as a confusing unzip error.
mpi_download() {
  local domain="$1" sfile="$2" out="$3"
  wget --post-data "username=$MPI_USER&password=$MPI_PASS" \
    "https://download.is.tue.mpg.de/download.php?domain=$domain&sfile=$sfile&resume=1" \
    -O "$out" --no-check-certificate --continue || true
  if unzip -tqq "$out" >/dev/null 2>&1; then
    return 0
  fi

  # The website host is not always the domain name: smplx -> smpl-x.
  local site="$domain"
  if [ "$domain" = "smplx" ]; then site="smpl-x"; fi
  echo ""
  if [ ! -s "$out" ]; then
    # 401: wget leaves a 0-byte file behind and writes nothing.
    echo "  !!! $sfile was refused (bad username or password)."
  else
    # 200 with an HTML body: usually the license was never accepted.
    echo "  !!! $sfile came back as something other than a zip:"
    echo "      $(head -c 120 "$out" | tr -dc '[:print:]')"
    echo "      This usually means the license is not accepted yet."
  fi
  echo "      Log in and accept it at https://$site.is.tue.mpg.de/"
  rm -f "$out"
  return 1
}

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

if ask_step "Download the 5 GVHMR checkpoints?"; then
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
fi

# 2 - SMPL body models  (requires free registration at https://smpl.is.tue.mpg.de/)
echo ""
echo "2/6 - SMPL body models - registration required"
SMPL_DIR="$CKPT_DIR/body_models/smpl"
if [ -f "$SMPL_DIR/SMPL_NEUTRAL.pkl" ]; then
  echo "  ==> Already in place, skipping."
elif ask_manual_step "Download the SMPL body models?"; then
  mpi_credentials
  mkdir -p "$SMPL_DIR"
  # v1.1.0, NOT v1.0.0: the 1.0.0 archive only ships male and female, so the
  # neutral model the pipeline needs is missing from it.
  if mpi_download smpl "SMPL_python_v.1.1.0.zip" "$SMPL_DIR/SMPL_python.zip"; then
    unzip -qo "$SMPL_DIR/SMPL_python.zip" -d "$SMPL_DIR/tmp_extract"
    for pair in "neutral:NEUTRAL" "m:MALE" "f:FEMALE"; do
      src=$(find "$SMPL_DIR/tmp_extract" -iname "basicmodel_${pair%%:*}_lbs_*_v1.1.0.pkl" -print -quit)
      if [ -n "$src" ]; then
        mv "$src" "$SMPL_DIR/SMPL_${pair##*:}.pkl"
        echo "  ==> SMPL_${pair##*:}.pkl"
      fi
    done
    rm -rf "$SMPL_DIR/tmp_extract" "$SMPL_DIR/SMPL_python.zip"
  else
    echo "      Fall back to a manual download from https://smpl.is.tue.mpg.de/"
    echo "      and place the models at:"
    echo "      3-Body/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_{GENDER}.pkl"
    read -p "      Press Enter once files are in place "
  fi
fi

# 3 - SMPLX body models  (requires free registration at https://smpl-x.is.tue.mpg.de/)
echo ""
echo "3/6 - SMPLX body models - registration required"
SMPLX_DIR="$CKPT_DIR/body_models/smplx"
if [ -f "$SMPLX_DIR/SMPLX_NEUTRAL.npz" ]; then
  echo "  ==> Already in place, skipping."
elif ask_manual_step "Download the SMPLX body models?"; then
  mpi_credentials
  mkdir -p "$SMPLX_DIR"
  # Note the domain is smplx, while the website is smpl-x.is.tue.mpg.de.
  if mpi_download smplx "models_smplx_v1_1.zip" "$SMPLX_DIR/models_smplx.zip"; then
    unzip -qo "$SMPLX_DIR/models_smplx.zip" -d "$SMPLX_DIR/tmp_extract"
    find "$SMPLX_DIR/tmp_extract" -name "SMPLX_*.npz" -exec mv -t "$SMPLX_DIR" {} +
    for g in NEUTRAL MALE FEMALE; do
      if [ -f "$SMPLX_DIR/SMPLX_$g.npz" ]; then echo "  ==> SMPLX_$g.npz"; fi
    done
    rm -rf "$SMPLX_DIR/tmp_extract" "$SMPLX_DIR/models_smplx.zip"
  else
    echo "      Fall back to a manual download from https://smpl-x.is.tue.mpg.de/"
    echo "      and place the models at:"
    echo "      3-Body/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_{GENDER}.npz"
    read -p "      Press Enter once files are in place "
  fi
fi

# 3b - Share the neutral SMPLX model with the Rendering stage.
#      Rendering hardcodes the name SMPLX_NEUTRAL_2020.npz; the GVHMR copy is the
#      same model (identical geometry and shapedirs, plus unused vt/ft), so link
#      instead of downloading it twice.
RENDER_SMPLX_DIR="$REPO_DIR/Rendering/data/smplx_models/smplx"
RENDER_SMPLX="$RENDER_SMPLX_DIR/SMPLX_NEUTRAL_2020.npz"
mkdir -p "$RENDER_SMPLX_DIR"
if [ -L "$RENDER_SMPLX" ]; then
  echo "  ==> Rendering symlink already in place, skipping."
elif [ -f "$RENDER_SMPLX" ]; then
  echo "  ==> Rendering already has a real SMPLX_NEUTRAL_2020.npz, leaving it alone."
elif [ -f "$CKPT_DIR/body_models/smplx/SMPLX_NEUTRAL.npz" ]; then
  ln -sr "$CKPT_DIR/body_models/smplx/SMPLX_NEUTRAL.npz" "$RENDER_SMPLX"
  echo "  ==> Linked Rendering/data/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
else
  echo "  ==> SMPLX_NEUTRAL.npz missing, skipping the Rendering symlink."
fi

# 4 - SMIRK checkpoints
echo ""
echo "4/6 - SMIRK checkpoints"
if ask_step "Download the SMIRK checkpoints?"; then
  cd "$REPO_DIR/4-Face/smirk"
  mkdir -p trained_models assets

  dl_file "trained_models/SMIRK_em1.pt" \
    gdown "1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE" -O trained_models/SMIRK_em1.pt

  dl_file "assets/face_landmarker.task" \
    wget -q --show-progress \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
      -O assets/face_landmarker.task

  if [ -f "assets/FLAME2020/generic_model.pkl" ]; then
    echo "  ==> FLAME2020 already in place, skipping."
  elif ask_manual_step "Download FLAME2020?"; then
    mpi_credentials
    if mpi_download flame "FLAME2020.zip" "FLAME2020.zip"; then
      mkdir -p assets/FLAME2020
      unzip -qo FLAME2020.zip -d assets/FLAME2020/
      rm FLAME2020.zip
      echo "  ==> FLAME2020"
    fi
  fi

  dl_file "yolov8l_100e.pt" \
    gdown "1iHL-XjvzpbrE8ycVqEbGla4yc1dWlSWU" -O yolov8l_100e.pt

  dl_file "fer2013_model.pth" \
    gdown --folder "1Tp9QsLZAoFEckvSTPw1VTI6ynk2UgFit" -O "$REPO_DIR/4-Face/smirk"
fi

# 5 - FastSAM-x.pt
echo ""
echo "5/6 - FastSAM-x.pt"
if ask_step "Download FastSAM-x.pt?"; then
  dl_file "$REPO_DIR/6-Captions/FastSAM-x.pt" \
    wget -q --show-progress \
      "https://huggingface.co/CASIA-LMC-Lab/FastSAM/resolve/main/FastSAM-x.pt" \
      -O "$REPO_DIR/6-Captions/FastSAM-x.pt"
fi

# 6 - yolo11n-pose.pt  (auto-downloaded by ultralytics on first run)
echo ""
echo "6/6 - yolo11n-pose.pt - auto-downloaded by Ultralytics on first filter run."

echo ""
echo " Download complete."
echo " SMPL, SMPLX and FLAME are fetched with your MPI login; if any of them"
echo " failed, accept its license on the matching website and re-run this script."