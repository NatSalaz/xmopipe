#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gvhmr_jz
echo "Active env : $(conda info --envs)"

# Chemins lus depuis config.yml (sections body/face/merge)
FILTER_DIR=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['body']['input_dir'])")
BODY_NPZ=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['body']['output_dir'])")
FACE_NPZ=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['face']['output_dir'])")
INPUT_BODY=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['merge']['input_body_dir'])")
INPUT_FACE=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['merge']['input_face_dir'])")
OUTPUT_MERGED=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['merge']['output_merged_dir'])")
OUTPUT_PP=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['merge']['output_ppmerged_dir'])")

echo "Merging into the npzs everything that what we may have missed before (Process not finished for example)."
cd ../3-Body/GVHMR
python post_merge_bodies.py --input_root "$FILTER_DIR" --npz_root "$BODY_NPZ"
cd ../../4-Face/smirk
python post_merge_faces.py --input_root "$FILTER_DIR" --npz_root "$FACE_NPZ"

echo "Merging of npzs done"
cd ../../5-Merge
echo "Merging face and body + PP started. (Mainly resampling to target fps)"
cd mergepp
python fusion_pipeline.py --input_body "$INPUT_BODY" --input_face "$INPUT_FACE" --output_root "$OUTPUT_MERGED"
python postsmooth.py --npz-folder "$OUTPUT_MERGED" --output "$OUTPUT_PP"
cd ../..
echo "Merging and PP finished."
conda deactivate
