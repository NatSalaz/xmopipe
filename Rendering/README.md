# Rendering

Inspection tools for the two motion formats the pipeline produces: the SMPL-X `.npz` scenes from
step 5, and the HumanML3D 263D `.npy` from step 7. None of this is part of the pipeline itself.

## Setup

```bash
conda env create -f ../xmo-visu.yml
conda activate xmo-visu
```

`xmo-3d` works too. Rendering is offscreen (`PYOPENGL_PLATFORM=egl`), so you don't need a display,
but you do need an EGL-capable driver.

You also need the SMPL-X model at `data/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz`. Run
`download.sh` at the repo root after the SMPL-X registration step and it will symlink the GVHMR
copy for you. Run the scripts from `Rendering/`, otherwise pass `--model_folder`.

## Scripts

| Script | Output |
|---|---|
| [visu.py](visu.py) | mesh or skeleton video of one scene |
| [visu_skeleton.py](visu_skeleton.py) | skeleton video, more camera options |
| [visu_folder.py](visu_folder.py) | grid video of every `.npz` in a folder |
| [visu_glb.py](visu_glb.py) | animated `.glb` for Blender or three.js |
| [visu_image.py](visu_image.py) | single frame as an image - **broken**, see below |
| [debug_visu_anim.py](debug_visu_anim.py) | interactive viewer for `.npz` scenes |
| [debug_visu_263.py](debug_visu_263.py) | interactive viewer for 263D `.npy` |

Samples to try them on live in `render_example/`.

```bash
cd Rendering

python visu.py --input render_example/example.npz --output example.mp4
python visu.py --input render_example/example.npz --output skel.mp4 --skeleton

python visu_skeleton.py --npz_file render_example/example.npz --output_dir . 
    --output_file skel.mp4 --follow_0 --text_overlay "a person waves"

python visu_folder.py --npz_file render_example --output_dir . --folder --fps 30

python visu_image.py --npz_file render_example/example.npz --output_dir . --output_file test.jpg --frame 25 [--body number]

python visu_glb.py --input render_example/example.npz --output example.glb

python debug_visu_anim.py --npz render_example/example.npz
```

Two things to know. `visu.py` takes `--input/--output` while the others take
`--npz_file/--output_dir/--output_file`. And `visu.py --output` is a filename, not a directory
despite what its help says - the file always lands in the current folder.

For `visu_folder.py`: `--spacing_x/--spacing_z` space out the grid, `--no_loop` stops short
sequences instead of looping them, `--no_labels` hides the scene names.

## Viewing 263D motions

`debug_visu_263.py` has the same controls as `debug_visu_anim.py` - play/pause, prev/next, speed,
left-drag to orbit, right-drag to pan, wheel to zoom, shadows toggle.

```bash
# one motion
python debug_visu_263.py --npy render_example/example.npy

# overlay two motions (Just to remind, 263D starts at 0,0,0 so they will supeprose)
python debug_visu_263.py --npy render_example/example.npy render_example/example2.npy

# a model output, which needs normalization
python debug_visu_263.py --npy recon.npy --mean <dataset>/Mean.npy --std <dataset>/Std.npy
```

The files in `new_joint_vecs/` are stored un-normalized, so you only need `--mean/--std` for
model outputs.

263D vectors carry joint positions and nothing else, so you get a 22-joint skeleton, not a mesh.
Getting a mesh back would mean fitting SMPL onto the joints, which is far too slow to do live.

## NPZ format

One key per person (`body_0`, `body_1`, …), each a pickled dict, so read them with
`np.load(..., allow_pickle=True)` then `d["body_0"].item()`. For a T-frame scene:

```
model            ()          'smplx'
gender           ()          'neutral'
poses            (T, 165)   float32   SMPL-X 2020, 55 joints x 3 (axis-angle)
betas            (T, 10)    float32   body shape
trans            (T, 3)     float32   root translation
expressions      (T, 50)    float32   facial expression coefficients
face_shape       (T, 300)   float32
emotions         (T,)       '<U5'     label per frame, drives the overlay
emotions_conf    (T, 7)     float32
bbox_xyxy        (T, 4)     float32
face_bbox_xyxy   (T, 4)     float32
contacts_conf    (T, 6)     float32
flagged_frames   (T,)       bool
cam_transl       (T, 3)     float64
fps / original_fps / start / stop  ()  int64
```

The overlay reads `emotions`; pass `--no_emotion` to `visu_skeleton.py` if that field is missing
or unreliable.

## Layout

The `visu*.py` scripts are thin argparse wrappers around `render_utils/`: `scene_render.py` for a
single scene, `folder_render.py` for the grid, `glb_render.py` for the GLB export.
`image_render.py` is an older renderer nothing calls anymore.

Videos and images are gitignored, so rendering into the source folder won't dirty the tree.

## Known issue

[visu_image.py](visu_image.py) imports `render_utils.fast_render_CL`, which doesn't exist here, so
it dies on import. The function it wants, `render_single_frame_mesh`, is in
[render_utils/scene_render.py](render_utils/scene_render.py) - looks like a stale module name.
