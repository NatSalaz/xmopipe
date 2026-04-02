#!/bin/bash
# XmoPipe - Environment setup
#
# Run this script once to create the two conda environments.
# Requires: Anaconda/Miniconda, CUDA 12.1+, git

set -e

CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
    CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
source "$CONDA_SH"

# Environment 1: xmo-3d (steps 1-5)
 echo "1/2 - Creating xmo-3d (steps 1-5: Download, Filter, Body, Face, Merge)"
 conda env create -f xmo-3d.yml
 conda activate xmo-3d
 
 echo "Installing torch 2.3.0+cu121"
 pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
 
 echo "Installing GVHMR (editable)"
 cd 3-Body/GVHMR
 pip install -r requirements.txt
 pip install -e .
 
 echo "Installing CUDA toolkit headers into conda env"
 conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
 
 echo "Installing DPVO"
 cd third-party/DPVO
 pip install "torch-scatter==2.1.2+pt23cu121" -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
 pip install "numba==0.61.0" "pypose==0.7.2" "ninja==1.13.0"
 pip install -e .
 cd ../../../..
 
 echo "Installing pytorch3d from source (this takes ~10 minutes)"
 pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"
 
 echo "Pinning numpy and ultralytics to match reference environment"
 pip install "numpy==1.26.4" "ultralytics==8.3.75" "av==14.1.0" "imageio==2.37.0"
 
 conda deactivate
 
# Environment 2: xmo-llm (steps 6-6b)
echo "2/2 - Creating xmo-llm (steps 6-6b: Captions, Caption augmentation)"
conda env create -f xmo-llm.yml
conda activate xmo-llm

echo "Installing real nccl from conda-forge (pip nvidia-nccl-cu12 is a stub)"
conda install -c conda-forge "nccl>=2.19" -y

echo "Installing torch 2.5.1+cu121"
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo "Installing FastSAM and CLIP from source"
pip install "ftfy==6.3.1" "regex==2026.3.32"
pip install "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
pip install "git+https://github.com/CASIA-LMC-Lab/FastSAM.git@b4ed20c2fed75eadc5aa7d8b09fedd137b873b52" --no-deps
pip install "ultralytics==8.0.120"
pip install "wordcloud"
pip install "nltk"
pip install "tensorboard"

echo "Downloading spacy language model"
python -m spacy download en_core_web_sm

conda deactivate

echo ""
echo "Setup complete."
echo "xmo-3d  : steps 1-5 (+ Rendering)"
echo "xmo-llm : steps 6-6b"
echo ""
echo "Before running the pipeline:"
echo "1. Fill in your YouTube API key in config.yml"
echo "2. Set the THEME variable in ../run_pipeline.sh (optional)"
echo "3. Run: cd .. && ./run_pipeline.sh"
