#!/usr/bin/env bash
set -euo pipefail

# Find all model files larger than MIN_SIZE_MB in the given directory
# Extensions incluses : .pth .pt .pkl .ckpt .bin .safetensors .onnx

ROOT_DIR="${1:-.}"
MIN_SIZE_MB="${2:-5}"

find "$ROOT_DIR" -type f \( \
    -iname "*.pth" -o \
    -iname "*.pt" -o \
    -iname "*.pkl" -o \
    -iname "*.ckpt" -o \
    -iname "*.bin" -o \
    -iname "*.safetensors" -o \
    -iname "*.onnx" \
\) -size +${MIN_SIZE_MB}M -print0 \
| xargs -0 du -h \
| sort -hr
