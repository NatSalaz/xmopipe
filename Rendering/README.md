# Rendering

Visualization of the SMPL-X `.npz` files produced by step 5 of the pipeline
(`5-Merge/mergepp/videosPPmerged/`): mesh or skeleton videos, single frames, and
animated GLB export.

Nothing here is part of the pipeline itself — these are inspection tools.

## Environment

```bash
conda env create -f ../xmo-visu.yml
conda activate xmo-visu
```

`xmo-3d` also works (it covers steps 1–5 plus rendering). Rendering is offscreen:
every entry point sets `PYOPENGL_PLATFORM=egl` before importing anything, so a GPU
without a display works, but an EGL-capable driver is required.

## SMPL-X model (required)

All renderers need `data/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz`. The filename is
hardcoded in `render_utils/*.py`.

`download.sh` (step 3b) symlinks it to the GVHMR copy at
`3-Body/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` — same model.
Run it after the SMPL-X registration step rather than placing the file by hand. See
[data/smplx_models/smplx/add_SMPLX_NEUTRAL_2020npz.txt](data/smplx_models/smplx/add_SMPLX_NEUTRAL_2020npz.txt).

Every renderer takes `--model_folder` (default `data/smplx_models`), so the scripts
must be run from `Rendering/` unless you pass an absolute path.

## Entry points

| Script | Output | Backend |
|---|---|---|
| [visu.py](visu.py) | mesh or skeleton video of one scene | `render_utils/scene_render.py` |
| [visu_skeleton.py](visu_skeleton.py) | skeleton video, more camera options | `render_utils/scene_render.py` |
| [visu_folder.py](visu_folder.py) | grid video of every `.npz` in a folder | `render_utils/folder_render.py` |
| [visu_glb.py](visu_glb.py) | animated `.glb` (Blender, three.js…) | `render_utils/glb_render.py` |
| [visu_image.py](visu_image.py) | single frame as an image | broken import, see below |
| [debug_visu_anim.py](debug_visu_anim.py) | interactive Open3D viewer | standalone (`open3d`, `smplx`) |

### Examples

```bash
cd Rendering

# mesh video
python visu.py --input render_example/example.npz --output example.mp4

# skeleton only
python visu.py --input render_example/example.npz --output example_skeleton.mp4 --skeleton

# skeleton with camera control and a caption burnt in
python visu_skeleton.py --npz_file render_example/example.npz --output_dir . \
    --output_file skel.mp4 --follow_0 --text_overlay "a person waves"

# every scene of a folder, side by side
python visu_folder.py --npz_file render_example --output_dir . --folder \
    --resolution 1920x1080 --fps 30

# animated GLB
python visu_glb.py --input render_example/example.npz --output example.glb --max_frames 60

# interactive viewer (needs a display, unlike the others)
python debug_visu_anim.py --npz render_example/example.npz
```

`visu.py` uses `--input/--output`; the other scripts use `--npz_file/--output_dir/--output_file`.
The inconsistency is historical.

Careful with `visu.py --output`: despite its help text ("Directory to save the output videos") it
is a **filename**, and the output directory is hardcoded to `.` at
[visu.py:40](visu.py#L40) — the file always lands in the current directory. Pass
`--output name.mp4`, not a folder.

Useful `visu_folder.py` flags: `--spacing_x/--spacing_z` (5.0) to space the grid,
`--cam_elevation` (0.6) / `--cam_distance` (1.3), `--no_loop` to stop short sequences
instead of looping them to the longest, `--no_labels` to drop the scene-name overlays.

## Expected NPZ format

One key per person in the scene (`body_0`, `body_1`, …), each a pickled dict — so
`np.load(..., allow_pickle=True)` and `d["body_0"].item()`. For a 74-frame scene:

```
model            ()          'smplx'
gender           ()          'neutral'
poses            (74, 165)   float32   SMPL-X 2020, 55 joints x 3 (axis-angle)
betas            (74, 10)    float32   body shape
trans            (74, 3)     float32   root translation
expressions      (74, 50)    float32   facial expression coefficients
face_shape       (74, 300)   float32
emotions         (74,)       '<U5'     label per frame, drives the overlay
emotions_conf    (74, 7)     float32
bbox_xyxy        (74, 4)     float32
face_bbox_xyxy   (74, 4)     float32
contacts_conf    (74, 6)     float32
flagged_frames   (74,)       bool
cam_transl       (74, 3)     float64
fps / original_fps / start / stop  ()  int64
```

The emotion overlay reads `emotions`; disable it with `--no_emotion` on
`visu_skeleton.py` when the field is absent or unreliable.

Note this is **not** the 263D HumanML3D format used for auto-encoder training —
that one is produced later, by step 7, and visualized from
[../training_SMPL_ae/](../training_SMPL_ae/) instead.

## Contents

```
visu*.py            CLI entry points (thin argparse wrappers)
render_utils/
  scene_render.py     one scene: render_multi_person_with_overlay,
                      ..._skeleton, render_single_frame_mesh
  folder_render.py    grid over a folder: render_folder_grid,
                      load_all_scenes_from_folder
  image_render.py     older single-scene renderer, currently unused
  glb_render.py       export_glb_animation
render_example/     sample NPZs (example.npz is the one kept in git)
data/smplx_models/  SMPL-X model goes here (gitignored)
```

`.mp4`, `.png` and `.npz` are gitignored, with `render_example/example.npz` explicitly
kept — rendering into the source folder will not dirty the tree.

## Known issue

[visu_image.py:10](visu_image.py#L10) imports `render_utils.fast_render_CL`, a module
that does not exist in this repo, so the script fails immediately on import. The
function it wants, `render_single_frame_mesh`, does exist in
[render_utils/scene_render.py:510](render_utils/scene_render.py#L510) — the import
appears to just point at a stale module name.

`image_render.py` is likewise no longer referenced by any entry point; `scene_render.py`
superseded it.
