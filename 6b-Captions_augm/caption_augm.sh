#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen

start_time=$(date +%s)

python cap_augm_Q34BFP8.py

AUGM_DIR=$(python -c "import yaml; c=yaml.safe_load(open('../config.yml')); print(c['captions_augm']['output_dir'])")
python add_POS.py --input "$AUGM_DIR"

end_time=$(date +%s)
echo "Caption augmentation executed in $((end_time - start_time)) seconds."

conda deactivate
