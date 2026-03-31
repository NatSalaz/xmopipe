#!/bin/bash
#SBATCH -A cua@v100
#SBATCH --job-name=test_ns
#SBATCH --output=logs/test/%j.out
#SBATCH --error=logs/test/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00

module load miniforge/24.9.0
module load cuda/12.1.0
echo "Noeud(s) assigne(s) : ${SLURM_NODELIST}"
# source miniconda
source $CONDA_PREFIX/bin/activate

echo "Activating the 'gvhmr' environment..."
conda activate gvhmr
# Echo actif environment to debug
echo "Environnement actif : $(conda info --envs)"

# Echo commands
start_time=$(date +%s)
# Check and automatically activate conda if necessary
# --- Run the first script with the 'gvhmr' environment ---
cd GVHMR
echo "Running demo_folder_pipeline.py..."
CUDA_LAUNCH_BLOCKING=1 python gvhmr_verif.py --input_root ../filteredVideos0605 --output_root out_body0605 --verbose
cd ..
echo "GVHMR execution completed."
conda deactivate
end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "Body recons pipeline have been successfully executed in $total_time seconds."
# Désactiver l'environnement après exécution

module unload miniforge/24.9.0
module unload cuda/12.1.0
