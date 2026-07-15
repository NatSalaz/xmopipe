# training_SMPL_ae

Entraînement, évaluation et exploration d'**auto-encodeurs de mouvement** sur des représentations SMPL / HumanML3D (vecteurs de pose 263-D, format `new_joint_vecs`).

Quatre familles de modèles partagent le même pipeline de données, d'entraînement et de losses :

| `model_type` | Modèle | Espace latent |
|--------------|--------|---------------|
| `vae`    | VAE dense                          | vecteur continu |
| `klvae`  | Auto-encodeur KL convolutionnel    | carte latente `(z_channels, T/down)` continue |
| `vqvae`  | VQ-VAE                             | tokens discrets (codebook) |
| `rvqvae` | Residual VQ-VAE                    | tokens discrets multi-quantizers |

Tout est piloté par des configs YAML (`configs/<dataset>/<model>.yaml`) ; le choix du modèle, du dataset et des losses en découle.

## Structure

```
train.py            # entraînement principal (config-driven)
train_ar.py         # variante (ancienne signature de modèles)
eval.py             # métriques de reconstruction (MPJPE / PA-MPJPE / ACCL)
latentvisu.py       # viewer PyQt5 de l'espace latent  ← outil interactif
latentspacevisu.py  # ancien script de visu latente (standalone)
configs/            # <dataset>/<model>.yaml
models/             # vae · klvae · vqvae · rvqvae + évaluateurs t2m
training/           # trainer.py, loss_manager.py, losses/
data/motion_loader.py  # MotionDataset + DATALoader (fenêtrage, normalisation)
latentspace/        # backend du viewer (encodage, PCA/t-SNE/UMAP, rendu Open3D)
common/ utils/      # squelette, quaternions, recover_from_ric, métriques
experiments/        # sorties d'entraînement (checkpoints, logs, tensorboard)
```

## Données

`MotionDataset` (dans [data/motion_loader.py](data/motion_loader.py)) charge des `.npy` de pose depuis `new_joint_vecs/`, découpe des fenêtres de `window_size` frames (64 par défaut) et normalise via `Mean.npy` / `Std.npy`. Les chemins des datasets (`t2m`, `xmo`, `idea400`, `hml3dxmo`, …) sont **codés en dur** dans le fichier — à adapter à la machine. On peut aussi passer `data_root` dans la config pour surcharger.

## Commandes

### Entraîner

```bash
python train.py --config configs/hml3d/klvae.yaml
```

Options utiles :

```bash
python train.py --config configs/hml3d/rvqvae.yaml \
    --exp-suffix run1 \        # suffixe du nom d'expérience
    --batch-size 128 \         # override de la config
    --latent-dim 256 \
    --resume experiments/<exp>/checkpoints/latest.pth
```

Sorties dans `experiments/<model>_<dataset>[_<suffix>]/` : `config.yaml`, `checkpoints/` (`latest.pth`, `best.pth`, `step_*.pth`) et logs TensorBoard (`tensorboard --logdir experiments`).

### Évaluer

Métriques de reconstruction (denormalise, `recover_from_ric` → joints 3D, puis MPJPE / PA-MPJPE / erreur d'accélération) :

```bash
python eval.py --config configs/hml3d/klvae.yaml \
    --checkpoint experiments/klvae_t2m/checkpoints/best.pth \
    --dataset t2m --batch-size 256
```

> Les métriques perceptuelles (FID / diversité) sont présentes mais commentées : on travaille sur des fenêtres courtes de 64 frames.

### `latentvisu` — explorer l'espace latent

```bash
python latentvisu.py --config configs/hml3d/klvae.yaml \
    --checkpoint experiments/klvae_t2m/checkpoints/best.pth \
    --dataset t2m
```

Application **PyQt5 + Open3D** ([latentvisu.py](latentvisu.py) → [latentspace/viewer_app.py](latentspace/viewer_app.py)) qui :

1. encode le split `test` du dataset pour construire un pool de latents ;
2. le projette en **2D** via trois onglets : **PCA**, **t-SNE** et **UMAP** ;
3. sur clic d'un point, décode le mouvement et l'**anime en 3D** (squelette Open3D) ;
4. affiche en surimpression les **k plus proches voisins** (slider `K`, interpolation pondérée des latents pour `K>1`) et les mouvements originaux ;
5. boutons pour afficher/masquer la **reconstruction** (vert) et les **originaux** (rouge).

Le décodage s'adapte au type de modèle (indices de codebook pour vq/rvqvae, carte/vecteur latent continu pour vae/klvae). Réglages par défaut du viewer (dataset, sous-échantillonnage, perplexité t-SNE…) dans [latentspace/config.py](latentspace/config.py).

## Notebooks

`reconstruction_klvae.ipynb` et `reconstruction_rvqvae.ipynb` : inspection visuelle des reconstructions par modèle.

## Dépendances

`torch`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `PyQt5`, `open3d`, `scikit-learn` (PCA/t-SNE) et `umap-learn` (projection UMAP, lancée en sous-processus).
