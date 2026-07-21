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

`MotionDataset` (in [data/motion_loader.py](data/motion_loader.py)) loads pose `.npy` files from `new_joint_vecs/`, slices windows of `window_size` frames (64 by default) and normalizes with `Mean.npy` / `Std.npy`.

### Where the datasets live

Dataset locations come from the `training` section of the repo-root [config.yml](../config.yml):

```yaml
training:
  dataset_dir: "./datasets"      # relative to the repo root
  dataset_dirs:
    t2m: "HumanML3D/HumanML3D"
    xmo: "XmoPipe/XmoPipe"
    ...
```

`dataset_dir` ships as a **symlink** to wherever the data actually sits (the datasets total ~200 GB, so don't copy them into the repo):

```bash
ln -s /path/to/your/datasets <repo-root>/datasets
```

A `dataset_name` in `configs/<dataset>/<model>.yaml` is looked up in `dataset_dirs`; setting `data_root` in that same config still overrides everything. To add a dataset, add one line to `dataset_dirs` — no Python to edit.

Each dataset folder must contain:

```
new_joint_vecs/*.npy      # 263D vectors — what training actually reads
texts/*.txt               # (`texts:` key in dataset_dirs if named otherwise)
Mean.npy, Std.npy         # normalization stats
train.txt val.txt test.txt all.txt   # split files, one motion id per line
```

### Normalization

`Mean.npy` / `Std.npy` are read from the dataset folder itself — `<dataset_dir>/<dataset>/Mean.npy`, next to `new_joint_vecs/` — and applied per-window in `__getitem__` as `(x - mean) / std`. `inv_transform()` reverses it, and is called before `recover_from_ric` wherever 3D joints are needed — [utils/evaluate_klvae.py](utils/evaluate_klvae.py), [utils/eval_t2m.py](utils/eval_t2m.py), [latentspacevisu.py](latentspacevisu.py).

There is **no** stats-computation step in this repo, and none is needed: the 263D representation is identical across all datasets, so HumanML3D's `Mean.npy` / `Std.npy` are copied into each dataset folder and reused as-is. A dataset without them will fail on `np.load` at construction time.

Consequence worth knowing: a model trained on one dataset can be evaluated or decoded on another without renormalizing, since they all share the same stats. The latent viewer relies on this ([latent_manager.py:60-61](latentspace/latent_manager.py#L60-L61) loads the stats of the *training* dataset, not of the one being displayed).

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

Training is optional: pretrained weights for the four model families are on the [project drive](../README.md#data), laid out in exactly this structure. Unpack them into `experiments/` and go straight to `eval.py` or `latentvisu.py`, passing the `config.yaml` that ships with each experiment.

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
