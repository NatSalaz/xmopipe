#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate smirk2
echo "Active env : $(conda info --envs)"
cd smirk

# Chemins lus depuis config.yml (section face.input_dir / face.output_dir)
INPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['face']['input_dir'])")
OUTPUT_ROOT=$(python -c "import yaml; c=yaml.safe_load(open('../../config.yml')); print(c['face']['output_dir'])")

echo "SMIRK launched"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
python smirk_verif_res.py --input_root "$INPUT_ROOT" --output_root "$OUTPUT_ROOT" --verbose
cd ..
echo "SMIRK finished."
conda deactivate