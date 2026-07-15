# training_SMPL_ae

Training, evaluation and exploration of **motion auto-encoders** on SMPL / HumanML3D representations (263-D pose vectors, `new_joint_vecs` format).

Four model families share the same data, training and loss pipeline:

| `model_type` | Model | Latent space |
|--------------|-------|--------------|
| `vae`    | Dense VAE                       | continuous vector |
| `klvae`  | Convolutional KL auto-encoder   | continuous latent map `(z_channels, T/down)` |
| `vqvae`  | VQ-VAE                          | discrete tokens (codebook) |
| `rvqvae` | Residual VQ-VAE                 | discrete tokens, multi-quantizer |

Everything is driven by YAML configs (`configs/<dataset>/<model>.yaml`); the model, dataset and losses all follow from it.

## Structure

```
train.py            # main training (config-driven)
train_ar.py         # variant (older model signatures)
eval.py             # reconstruction metrics (MPJPE / PA-MPJPE / ACCL)
latentvisu.py       # PyQt5 latent-space viewer  ← interactive tool
latentspacevisu.py  # older standalone latent-visu script
configs/            # <dataset>/<model>.yaml
models/             # vae · klvae · vqvae · rvqvae + t2m evaluators
training/           # trainer.py, loss_manager.py, losses/
data/motion_loader.py  # MotionDataset + DATALoader (windowing, normalization)
latentspace/        # viewer backend (encoding, PCA/t-SNE/UMAP, Open3D rendering)
common/ utils/      # skeleton, quaternions, recover_from_ric, metrics
experiments/        # training outputs (checkpoints, logs, tensorboard)
```

## Data

`MotionDataset` (in [data/motion_loader.py](data/motion_loader.py)) loads pose `.npy` files from `new_joint_vecs/`, slices windows of `window_size` frames (64 by default) and normalizes with `Mean.npy` / `Std.npy`. Dataset paths (`t2m`, `xmo`, `idea400`, `hml3dxmo`, …) are **hard-coded** in the file — adapt them to your machine. You can also pass `data_root` in the config to override.

## Commands

### Train

```bash
python train.py --config configs/hml3d/klvae.yaml
```

Useful options:

```bash
python train.py --config configs/hml3d/rvqvae.yaml \
    --exp-suffix run1 \        # experiment-name suffix
    --batch-size 128 \         # config override
    --latent-dim 256 \
    --resume experiments/<exp>/checkpoints/latest.pth
```

Outputs land in `experiments/<model>_<dataset>[_<suffix>]/`: `config.yaml`, `checkpoints/` (`latest.pth`, `best.pth`, `step_*.pth`) and TensorBoard logs (`tensorboard --logdir experiments`).

### Evaluate

Reconstruction metrics (denormalize, `recover_from_ric` → 3D joints, then MPJPE / PA-MPJPE / acceleration error):

```bash
python eval.py --config configs/hml3d/klvae.yaml \
    --checkpoint experiments/klvae_t2m/checkpoints/best.pth \
    --dataset t2m --batch-size 256
```

> Perceptual metrics (FID / diversity) are present but commented out: we work on short 64-frame windows.

### `latentvisu` — explore the latent space

```bash
python latentvisu.py --config configs/hml3d/klvae.yaml \
    --checkpoint experiments/klvae_t2m/checkpoints/best.pth \
    --dataset t2m
```

A **PyQt5 + Open3D** app ([latentvisu.py](latentvisu.py) → [latentspace/viewer_app.py](latentspace/viewer_app.py)) that:

1. encodes the dataset `test` split to build a latent pool;
2. projects it to **2D** across three tabs: **PCA**, **t-SNE** and **UMAP**;
3. on clicking a point, decodes the motion and **animates it in 3D** (Open3D skeleton);
4. overlays the **k nearest neighbors** (`K` slider, weighted latent interpolation for `K>1`) and the original motions;
5. buttons to show/hide the **reconstruction** (green) and the **originals** (red).

Decoding adapts to the model type (codebook indices for vq/rvqvae, continuous latent map/vector for vae/klvae). Viewer defaults (dataset, subsampling, t-SNE perplexity…) live in [latentspace/config.py](latentspace/config.py).

## Notebooks

`reconstruction_klvae.ipynb` and `reconstruction_rvqvae.ipynb`: visual inspection of per-model reconstructions.

## Dependencies

`torch`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `PyQt5`, `open3d`, `scikit-learn` (PCA/t-SNE) and `umap-learn` (UMAP projection, run in a subprocess).
