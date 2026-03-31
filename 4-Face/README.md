# theseGit

### SMIRK

Réutilise SMIRK et ResEmoteNet pour qu'à partir d'une vidéo obtenir les données nécessaires pour la suite de la pipeline.

SMIRK arXiv: arXiv:2404.04104
ResEmoteNet DOI: arXiv:2409.10545

python smirk_verif_res.py --input_root ../../2-Filter/filteredVideos --output_root ./out_face

conda activate smirk
cd smirk
python smirk_verif_res.py --input_root ../../2-Filter/filteredVideos --output_root ./out_face
cd ..


Les outputs voulus sont les .npz "i_face" où i est le numéro de vidéo.