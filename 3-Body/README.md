# theseGit

### GVHMR arxiv/2409.06662

Réutilise la démo de GVHMR pour qu'à partir d'une vidéo obtenir les données nécessaires pour la suite de la pipeline.
demo_folder_pipeline.py récupère une architecture de fichiers comme celle de filteredVideos et la traite.

GVHMR arXiv: https://arxiv.org/abs/2409.06662

GVHMR:
python demo_multi_smplx.py  --video ../video.mp4 --output_root output/ (--verbose pour de la verbose et --skeleton pour avoir les vidéos des squelettes 2D et bounding boxes)

conda activate gvhmr
cd GVHMR
CUDA_LAUNCH_BLOCKING=1 python gvhmr_verif.py --input_root ../../2-Filter/filteredVideos --output_root ./out_body --max_frames 500 --verbose
cd ..
conda deactivate

Les outputs voulus sont les .npz "i_body" où i est le numéro de vidéo.