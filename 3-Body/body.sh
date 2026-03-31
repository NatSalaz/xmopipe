#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gvhmr_jz
echo "Active env : $(conda info --envs)"
cd GVHMR

# Chemins lus depuis config.yml (section body.input_dir / body.output_dir)
INPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['body']['input_dir'])")
OUTPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['body']['output_dir'])")

echo "GVHMR launched"
CUDA_LAUNCH_BLOCKING=1 python gvhmr_verif.py --input_root "$INPUT_ROOT" --output_root "$OUTPUT_ROOT" --verbose
cd ..
echo "GVHMR finished."

conda deactivate