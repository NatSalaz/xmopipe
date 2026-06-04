#!/bin/bash
#  XmoPipe - Full pipeline bash script
#
#  Steps 3 (Body) and 4 (Face) run sequentially on the same GPU but they could theorically be parallel though.
#  Step 7 (Conversion) is a Jupyter notebook — run manually.

ENV_3D="xmo-3d"    # Steps 1-5: GVHMR + SMIRK + Filter
ENV_LLM="xmo-llm"  # Steps 6-6b: Qwen3-VL + Qwen3-4B

# Theme for YouTube query generation (step 1).
# Leave empty to skip and use existing video_ideas.txt.

THEME="yoga"               # e.g. THEME="yoga"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

source ~/anaconda3/etc/profile.d/conda.sh

echo "XmoPipe Pipeline"
echo "Repo : $REPO_DIR"

# Step 1 - Download 
conda activate "$ENV_3D"

echo "1 - Download (yt-dlp + scene cut)"
cd "$REPO_DIR/1-Download"
if [ -n "$THEME" ]; then
  echo "Generating query ideas for theme: $THEME"
  python YTPromptIdeas.py "$THEME"
fi
python YTdl.py --verbose
python YTcut.py --keep

# Step 2 - Filter 
echo "2 - Filter (YOLO + optical flow)"
cd "$REPO_DIR/2-Filter"
INPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['filter']['input_dir'])")
OUTPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['filter']['output_dir'])")
python YTfilter_ultra.py --input_root "$INPUT_ROOT" --output_root "$OUTPUT_ROOT"

# Step 3 - Body 
echo "3 - Body (GVHMR)"
cd "$REPO_DIR/3-Body/GVHMR"
INPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['body']['input_dir'])")
OUTPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['body']['output_dir'])")
CUDA_LAUNCH_BLOCKING=1 python gvhmr_verif.py \
  --input_root "$INPUT_ROOT" --output_root "$OUTPUT_ROOT" --verbose

# Step 4 - Face 
echo "4 - Face (SMIRK)"
cd "$REPO_DIR/4-Face/smirk"
INPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['face']['input_dir'])")
OUTPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['face']['output_dir'])")
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
python smirk_verif_res.py \
  --input_root "$INPUT_ROOT" --output_root "$OUTPUT_ROOT" --verbose

# Step 5 - Merge 
echo "5 - Merge (face + body fusion, post-processing)"

cd "$REPO_DIR/3-Body/GVHMR"
FILTER_DIR=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['body']['input_dir'])")
BODY_NPZ=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['body']['output_dir'])")
python post_merge_bodies.py --input_root "$FILTER_DIR" --npz_root "$BODY_NPZ"

cd "$REPO_DIR/4-Face/smirk"
FACE_NPZ=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['face']['output_dir'])")
python post_merge_faces.py --input_root "$FILTER_DIR" --npz_root "$FACE_NPZ"

cd "$REPO_DIR/5-Merge/mergepp"
INPUT_BODY=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['merge']['input_body_dir'])")
INPUT_FACE=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['merge']['input_face_dir'])")
OUTPUT_MERGED=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['merge']['output_merged_dir'])")
OUTPUT_PP=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['merge']['output_ppmerged_dir'])")
python fusion_pipeline.py \
  --input_body "$INPUT_BODY" --input_face "$INPUT_FACE" --output_root "$OUTPUT_MERGED"
python postsmooth.py --npz-folder "$OUTPUT_MERGED" --output "$OUTPUT_PP"

conda deactivate

# Step 6 - Captions 
echo "6 - Captions (Qwen3-VL-8B)"
conda activate "$ENV_LLM"
cd "$REPO_DIR/6-Captions"
VIDEO_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['captions']['video_root'])")
NPZ_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['captions']['npz_root'])")
python vcap3VL8B.py --video_root "$VIDEO_ROOT" --npz_root "$NPZ_ROOT"

conda deactivate

echo "Pipeline complete."
echo  "For step 7: open 7-Conversion_263/raw_pose_processing.ipynb"