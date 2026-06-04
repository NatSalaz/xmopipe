# XmoPipe

Multi-stage motion capture pipeline from YouTube videos, producing a dataset in HumanML3D format (263D).
Accepted at CASAXR 2026

---

## Data

Data is available at this link: `https://drive.google.com/drive/folders/1Tp9QsLZAoFEckvSTPw1VTI6ynk2UgFit?usp=sharing`

NPZ data is defined like this: 

```
video_XXXX
├── description_videos_video_XXXX.json
├── metadata.txt
├── video_XXXX_merged_scene_1.npz
├── video_XXXX_merged_scene_2.npz
├── video_XXXX_merged_scene_3.npz
├── video_XXXX_merged_scene_4.npz
├── video_XXXX_merged_scene_5.npz
```
JSONs contains the textual descriptions for each video
metadata.txt contains video link and each scene timestamp.

NPZs contain extracted infos: 
They contain 1 key namex 'body_Y' for each body each containing the following keys:

```
[model] | Type: str | Example: 'smplx2020'
[expressions] | Type: numpy.ndarray | Example: shape: (77, 50)
[trans] | Type: numpy.ndarray | Example: shape: (77, 3)
[betas] | Type: numpy.ndarray | Example: shape: (77, 10)
[poses] | Type: numpy.ndarray | Example: shape: (77, 165)
[gender] | Type: str | Example: 'neutral'
[cam_transl] | Type: numpy.ndarray | Example: shape: (77, 3)
[emotions] | Type: numpy.ndarray | Example: shape: (77,)
[emotions_conf] | Type: numpy.ndarray | Example: shape: (65, 7)
[face_bbox_xyxy] | Type: numpy.ndarray | Example: shape: (65, 4)
[flagged_frames] | Type: numpy.ndarray | Example: shape: (77,)
[bbox_xyxy] | Type: numpy.ndarray | Example: shape: (77, 4)
[fps] | Type: int | Example: 30
[original_fps] | Type: int | Example: 25
[start] | Type: int | Example: 0
[stop] | Type: int | Example: 77
```

NPYs contain data on HumanML3D format ==> 263D vectors and texts corresponding. The chosen texts associated with the motion are the texts contained in `<Action>` tags

## Installation

### Tested on

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | Linux 6.17.0-19-generic |
| GPU | NVIDIA RTX 2000 Ada Generation 8 GB |
| CPU | Intel Core Ultra 7 165H |
| CUDA | 12.1 |

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or Miniconda
- ffmpeg (`sudo apt install ffmpeg`)
- [Ollama](https://ollama.com/) for the optional step 1 query generation (`YTPromptIdeas.py`)

### Quick install

```bash
git clone https://github.com/NatSalaz/xmopipe.git
cd xmopipe
bash setup.sh      # creates xmo-3d and xmo-llm conda environments (~30 min)
bash download.sh   # downloads all model checkpoints (~10 GB)
```

`setup.sh` creates two environments:
- `xmo-3d` — steps 1–5  + Rendering (Python 3.10, torch 2.3.0+cu121, pytorch3d)
- `xmo-llm` — steps 6–6b (Python 3.11, torch 2.5.1+cu121)

`download.sh` downloads checkpoints for GVHMR, SMIRK, FastSAM, and ResEmoteNet into the expected paths. Files already present are skipped.

### YouTube API key

Step 1 uses the YouTube Data v3 API (free, 100 requests/day).
```
1. Generate a key at [Google Cloud Console](https://console.cloud.google.com/) - APIs & Services - Credentials

2. Copy your key in config.yml file in the corresponding section

3. We advise to use a google account on firefox, and not abuse this script, otherwise you will get ip banned on your account by YouTube.
```

---
## Quick start

### All parameters are centralized in [`config.yml`](config.yml) at the root of the repo. Edit it before running anything.

Run the steps in the following order. Steps 3 and 4 are independent and can run in parallel across multiple machines - each script uses a file-based locking system (`video_verif.py`) to avoid conflicts.

```bash
cd 1-Download        
./download.sh
cd 2-Filter          
./filter.sh
cd 3-Body            
./body.sh       # can run in parallel with step 4 on a separate machine
cd 4-Face            
./face.sh       # can run in parallel with step 3 on a separate machine
cd 5-Merge           
./mergePP.sh    # needs steps 3 and 4 to be complete
cd 6-Captions        
./caption.sh    # needs step 5
cd 6b-Captions_augm  
./caption.sh    # needs step 6
# Step 7: open 7-Conversion_263/raw_pose_processing.ipynb and run cells in order (needs steps 5 and 6)
```
## Pipeline steps

### 1 - Download

Downloads and cuts videos into scenes.

- `YTPromptIdeas.py` - generates search query ideas using a local LLM (Ollama)
- `YTdl.py` - downloads videos via yt-dlp
- `YTcut.py` - cuts scenes using PySceneDetect

Output: `1-Download/cutVideos/`

```bash
./download.sh
```

Or step by step:
```bash
python YTPromptIdeas.py <video_theme>
python YTdl.py --verbose
python YTcut.py [--keep] [--min-duration <seconds>] [--input-dir <dir>] [--output-dir <dir>]
```

### 2 - Filter

Filters and re-cuts the scenes from step 1. Filtering criteria include optical flow, presence and size of detected persons, bounding box quality, crowd detection, and frozen frame detection.

Some flags are written into the `metadata.txt` files when persons have positions where the feet appear above the head in 2D, or when multiple persons are detected. These flags are not reused elsewhere in the pipeline but are available for analysis.

Output: `2-Filter/filteredVideos/`

```bash
./filter.sh
```

Or:
```bash
python YTfilter_ultra.py --input_root <input path> --output_root <output path> \
  [--min_bbox_area <area>] [--max_segment_length <frames>] [--keep] [--verbose] [--skeleton]
```

### 3 - Body

Estimates global body pose using GVHMR (HMR4D). Also outputs camera translation and 2D bounding boxes reused in steps 5 and 6.

GVHMR: [arXiv:2409.06662](https://arxiv.org/abs/2409.06662)

Input: `2-Filter/filteredVideos/` - Output: `3-Body/GVHMR/out_body/`

```bash
./body.sh
```

Or:
```bash
cd 3-Body/GVHMR
CUDA_LAUNCH_BLOCKING=1 python gvhmr_verif.py \
  --input_root <input path> --output_root <output path> \
  [--static_cam] [--verbose] [--force_reprocess] [--max_frames <n>]
```

### 4 - Face

Extracts facial expression vectors (SMIRK/FLAME) and per-frame emotion labels (ResEmoteNet).

SMIRK: [arXiv:2404.04104](https://arxiv.org/abs/2404.04104) - ResEmoteNet: [DOI:10.1109/LSP.2024.3521321](https://doi.org/10.1109/LSP.2024.3521321)

Input: `2-Filter/filteredVideos/` - Output: `4-Face/smirk/out_face/`

```bash
./face.sh
```

Or:
```bash
cd 4-Face/smirk
python smirk_verif_res.py --input_root <input path> --output_root <output path> \
  [--smirk_checkpoint <path>] [--device <device>] [--batch_size <n>] [--max_frames <n>]
```

### 5 - Merge

Fuses the face and body data from steps 3 and 4. `fusion_pipeline.py` assigns faces to bodies using nose and eye landmark distances, shifts translations relative to camera motion, and centres the scene. `postsmooth.py` resamples everything to 30fps, applies Gaussian smoothing on expressions and jaw pose, smooths emotions via a sliding majority vote window, removes data islands, and filters out static sequences based on rotation and jerk thresholds.

Input: `3-Body/GVHMR/out_body/` + `4-Face/smirk/out_face/` - Output: `5-Merge/mergepp/videosPPmerged/`

```bash
./mergePP.sh
```

Or, if some NPZs were not merged correctly during steps 3/4:
```bash
cd 3-Body/GVHMR 
python post_merge_bodies.py --input_root <videos path> --npz_root <body npzs path>
cd 4-Face/smirk 
python post_merge_faces.py --input_root <videos path> --npz_root <face npzs path>
```
Then:
```bash
cd 5-Merge/mergepp
python fusion_pipeline.py --input_body <body npzs> --input_face <face npzs> --output_root <output> [--no_smooth]
python postsmooth.py --npz-folder <merged folder> --output <output> [--smplx-model <path>] [--device <device>]
```

### 6 - Captions

Draws temporary person outlines with IDs on the videos (inspired by [arXiv:2410.02244](https://arxiv.org/abs/2410.02244)), then runs Qwen3-VL-8B to generate structured captions describing actions, body posture, and movement style for each visible person.

Input: `2-Filter/filteredVideos/` + `5-Merge/mergepp/videosPPmerged/` - Output: JSON files alongside NPZs

```bash
./caption.sh
```

Or:
```bash
cd 6-Captions
python vcap3VL8B.py --video_root <videos path> --npz_root <npz path> \
  [--model_path <path>] [--max_tokens <n>] [--max_frames <n>]
```

### 6b - Caption augmentation

Takes the raw captions from step 6 and uses Qwen3-4B to generate rephrased versions, then adds POS tags formatted for use with motion generation models.

Input: `5-Merge/mergepp/videosPPmerged/` - Output: `6b-Captions_augm/augm_txts/`

```bash
./caption.sh
```

Or:
```bash
cd 6b-Captions_augm
python cap_augm_Q34BFP8.py
python add_POS.py --input ./augm_txts [--output <output dir>]
```

### 7 - Conversion

Converts the merged NPZs into `.npy` files in HumanML3D 263D format.

HumanML3D: [DOI:10.1109/CVPR52688.2022.00509](https://doi.org/10.1109/CVPR52688.2022.00509)

In order to convert, you will need to
```
Open `7-Conversion_263` folder

Run the cells of `create_csv.ipynb` in order.
Run the cells of `raw_pose_processing.ipynb` in order.
Run the cells of `motion_representation.ipynb` in order.
Run the cells of `split_data.ipynb` in order.
```

You will obtain a dataset with this folder organization:
```
.
├── all.txt
├── metadatas
├── new_joints
│   ├── example_1_0.npy
│   └── example_2_0.npy
├── new_joint_vecs
│   ├── example_1_0.npy
│   └── example_2_0.npy
├── test.txt
├── texts
│   ├── example_1_0.txt
│   └── example_2_0.txt
├── train.txt
├── train_val.txt
└── val.txt
```
new_joints_vecs contains the 263D format. 
Texts associated in texts folder are the `<Action>` sections of our npzs.


Input: `5-Merge/mergepp/videosPPmerged/` + caption JSON files - Output: `7-Conversion_263/XmoPipe/`

**Example input structure:**
```
videosPPmerged/
├── video_X/
│   ├── metadata.txt
│   ├── description_videos_video_X.json
│   ├── video_X_merged_scene_1.npz   # 2 persons in this example
│   └── video_X_merged_scene_2.npz
└── video_Y/
    ├── metadata.txt
    ├── description_videos_video_Y.json
    └── video_Y_merged_scene_1.npz
```

**Example output structure** (X_2_0 is in the test split):
```
XmoPipe/
├── motion_data/smplx_322/
│   ├── dataset/
│   │   ├── X_1_0.npy
│   │   ├── X_1_1.npy   # person 1 (2-person scene)
│   │   ├── X_2_0.npy
│   │   └── Y_1_0.npy
│   ├── dataset_test_align/
│   │   └── X_2_0.npy
│   └── dataset_train_val_align/
│       ├── X_1_0.npy, X_1_1.npy, Y_1_0.npy
└── texts/semantic_labels/
    ├── dataset/
    │   ├── X_1_0.txt, X_1_1.txt, X_2_0.txt, Y_1_0.txt
    ├── dataset_test_align/
    │   └── X_2_0.txt
    └── dataset_train_val_align/
        ├── X_1_0.txt, X_1_1.txt, Y_1_0.txt
```

### 8 - Stats

Jupyter notebooks describing the dataset statistics.

### AE training

Training is done by using train.py:
`python train.py --config configs/xmo/rvqvae.yaml` for example to train a RVQ-VAE on xmopipe.
You can look into and modify configurations directly in the config files. 

In order to visualize a latent visualization of your saved experiments you will need to install:
```
pip install pyqt5
pip install scikit-learn
pip install umap-learn
```

Use example:
`python latentvisu.py --config experiments/klvae_hml3dxmoI400_ld256/config.yaml --checkpoint experiments/klvae_hml3dxmoI400_ld256/checkpoints/best.pth --dataset t2m`
In order to use this given configuration and checkpoint on the HumanML3D dataset

---

### Rendering

In order to render NPZs, you have the Rendering folder containing several scripts to visualize SMPL-X data.
An example file with 2 bodies is given in folder render_example.
Use:
```
python visu.py --input render_example/example.npz --output example.mp4
python visu.py --input render_example/example.npz --output example_skeleton.mp4 --skeleton
python debug_visu_anim.py --npz render_example/example.npz
```


## Result examples

| **Original** | **Outline for caption inference** |
|:---:|:---:|
| ![GIF 1](assets/3_original.gif) | ![GIF 2](assets/tmp_3.gif) |
| **Caption result** | **3D inference result** |
| ![GIF 3](assets/3_caption.gif) | ![GIF 4](assets/3D_3.gif) |

## Citation

If you use this repository in your research or build upon our pipeline, please cite the **core upstream works** that are fundamental to this project:

- **[SMIRK](https://github.com/georgeretsi/smirk)** for 3D facial expression reconstruction
- **[GVHMR](https://github.com/zju3dv/GVHMR)** for world-grounded human motion recovery
- **[Qwen3-VL](https://huggingface.co/Qwen)** for vision-language understanding components

```bibtex
@inproceedings{SMIRK:CVPR:2024,
    title = {3D Facial Expressions through Analysis-by-Neural-Synthesis},
    author = {Retsinas, George and Filntisis, Panagiotis P. and Danecek, Radek and Abrevaya, Victoria F. and Roussos, Anastasios and Bolkart, Timo and Maragos, Petros},
    booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2024}
}

@inproceedings{shen2024gvhmr,
    title = {World-Grounded Human Motion Recovery via Gravity-View Coordinates},
    author = {Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
    booktitle = {SIGGRAPH Asia Conference Proceedings},
    year = {2024}
}

@misc{bai2025qwen3vltechnicalreport,
    title = {Qwen3-VL Technical Report},
    author = {Shuai Bai and Yuxuan Cai and Ruizhe Chen and Keqin Chen and Xionghui Chen and Zesen Cheng and Lianghao Deng and Wei Ding and Chang Gao and Chunjiang Ge and Wenbin Ge and Zhifang Guo and Qidong Huang and Jie Huang and Fei Huang and Binyuan Hui and Shutong Jiang and Zhaohai Li and Mingsheng Li and Mei Li and Kaixin Li and Zicheng Lin and Junyang Lin and Xuejing Liu and Jiawei Liu and Chenglong Liu and Yang Liu and Dayiheng Liu and Shixuan Liu and Dunjie Lu and Ruilin Luo and Chenxu Lv and Rui Men and Lingchen Meng and Xuancheng Ren and Xingzhang Ren and Sibo Song and Yuchong Sun and Jun Tang and Jianhong Tu and Jianqiang Wan and Peng Wang and Pengfei Wang and Qiuyue Wang and Yuxuan Wang and Tianbao Xie and Yiheng Xu and Haiyang Xu and Jin Xu and Zhibo Yang and Mingkun Yang and Jianxin Yang and An Yang and Bowen Yu and Fei Zhang and Hang Zhang and Xi Zhang and Bo Zheng and Humen Zhong and Jingren Zhou and Fan Zhou and Jing Zhou and Yuanzhi Zhu and Ke Zhu},
    year = {2025},
    eprint = {2511.21631},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url = {https://arxiv.org/abs/2511.21631}
}
```
## Acknowledgements

This repository builds upon several outstanding open-source research projects and models:

- [GVHMR](https://github.com/zju3dv/GVHMR)
- [SMIRK](https://github.com/georgeretsi/smirk)
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- [SMPL-X](https://github.com/vchoutas/smplx)
- [Qwen3-VL](https://huggingface.co/Qwen)

We sincerely thank the authors and contributors of these projects for releasing their code, models, and research resources to the community. Their work made this repository possible.
