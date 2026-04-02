#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate xmo-3d
echo "Active env : $(conda info --envs)"
nvidia-smi
nvcc --version
python YTfilter_ultra.py --input_root ../1-Download/cutVideos/ --output_root ./filteredVideos/
conda deactivate
