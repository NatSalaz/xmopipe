#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate xmo-llm

# Paths read from config.yml (sections captions.video_root / captions.npz_root)
VIDEO_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['captions']['video_root'])")
NPZ_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['captions']['npz_root'])")

start_time=$(date +%s)

python vcap3VL8B.py --video_root "$VIDEO_ROOT" --npz_root "$NPZ_ROOT"

end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Caption executed in $total_time seconds."

conda deactivate