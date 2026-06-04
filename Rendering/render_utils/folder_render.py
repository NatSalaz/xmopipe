import os
import glob
import cv2
import torch
import numpy as np
import math
import trimesh
import pyrender
import smplx
from tqdm import tqdm
from joblib import Parallel, delayed


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def look_at(eye, target, up=np.array([0, 1, 0])):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_corrected = np.cross(right, forward)
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up_corrected
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


def project_point_pyrender(point_3d, view_matrix, projection_matrix, width, height):
    point = np.append(point_3d, 1.0)
    clip = projection_matrix @ (view_matrix @ point)
    if clip[3] == 0:
        return None
    ndc = clip[:3] / clip[3]
    px = int((ndc[0] + 1) * 0.5 * width)
    py = int((1 - ndc[1]) * 0.5 * height)
    return px, py


# ─────────────────────────────────────────────
# SMPL-X loader (single scene)
# ─────────────────────────────────────────────


def load_smplx_data(
    npz_file, model_folder, model_type, gender, num_betas, num_emotion_coeffs, device
):
    models = {}
    faces = None
    trans_dict = {}
    expressions_dict = {}
    emotions_dict = {}
    joints_dict = {}
    start_dict = {}
    stop_dict = {}

    scene_data = np.load(npz_file, allow_pickle=True)
    print(f"Loaded: {npz_file}")

    for person_id in scene_data.files:
        person_data = scene_data[person_id].item()
        start = person_data.get("start", None)
        stop = person_data.get("stop", None)

        required_keys = ["poses", "betas", "trans", "cam_transl"]
        missing = [k for k in required_keys if k not in person_data]
        if missing:
            print(f"  ⚠ {person_id}: missing {missing}, skipped")
            continue
        if "expressions" not in person_data and "emotions" not in person_data:
            print(f"  ⚠ {person_id}: no expressions/emotions, skipped")
            continue

        poses = person_data["poses"]
        expected_len = 165
        if poses.shape[1] < expected_len:
            poses = np.pad(
                poses, ((0, 0), (0, expected_len - poses.shape[1])), mode="constant"
            )
        elif poses.shape[1] > expected_len:
            poses = poses[:, :expected_len]

        trans = person_data["trans"]
        betas = person_data["betas"]
        pose_size = poses.shape[0]
        if betas.ndim == 1 or (betas.ndim == 2 and betas.shape[0] == num_betas):
            betas = np.tile(betas, (pose_size, 1))

        if faces is None:
            faces_path = os.path.join(model_folder, "smplx", "SMPLX_NEUTRAL_2020.npz")
            faces = np.load(faces_path, allow_pickle=True)["f"]

        expr = person_data.get("expressions", None)
        emot = person_data.get("emotions", None)

        model = smplx.create(
            model_folder,
            model_type=model_type,
            gender=gender,
            num_betas=num_betas,
            num_expression_coeffs=num_emotion_coeffs,
            use_pca=False,
        ).cuda()
        if expr is not None:
            expression = torch.tensor(expr).float().cuda()
        else:
            expression = torch.zeros(betas.shape[0], num_emotion_coeffs).float().cuda()
        output = model(
            betas=torch.tensor(betas).float().cuda(),
            transl=torch.tensor(trans).float().cuda(),
            expression=expression,
            global_orient=torch.tensor(poses[:, :3]).float().cuda(),
            body_pose=torch.tensor(poses[:, 3:66]).float().cuda(),
            jaw_pose=torch.tensor(poses[:, 66:69]).float().cuda(),
            leye_pose=torch.tensor(poses[:, 69:72]).float().cuda(),
            reye_pose=torch.tensor(poses[:, 72:75]).float().cuda(),
            left_hand_pose=torch.tensor(poses[:, 75:120]).float().cuda(),
            right_hand_pose=torch.tensor(poses[:, 120:165]).float().cuda(),
            return_verts=True,
        )

        key = f"{os.path.basename(npz_file)}__{person_id}"
        models[key] = output.vertices.cpu().numpy()
        joints_dict[key] = output.joints.cpu().numpy()
        trans_dict[key] = trans
        start_dict[key] = start
        stop_dict[key] = stop
        expressions_dict[key] = expr
        emotions_dict[key] = (
            emot
            if emot is not None
            else np.array([[f"{v:.2f}" for v in frame] for frame in expr])
        )

    if not models:
        raise ValueError(f"No valid person found in {npz_file}")

    return (
        models,
        faces,
        trans_dict,
        expressions_dict,
        emotions_dict,
        joints_dict,
        start_dict,
        stop_dict,
    )


# ─────────────────────────────────────────────
# Ground plane
# ─────────────────────────────────────────────


def add_simple_ground(scene, vertices_dict):
    all_v = np.concatenate([v[0] for v in vertices_dict.values()], axis=0)
    min_x, max_x = all_v[:, 0].min(), all_v[:, 0].max()
    min_z, max_z = all_v[:, 2].min(), all_v[:, 2].max()
    min_y = all_v[:, 1].min()  # Étendre légèrement la grille autour des personnages

    margin = 2.0
    min_x -= margin
    max_x += margin
    min_z -= margin
    max_z += margin
    cell = 1.0  # taille d'une case en mètres
    h = 0.01  # épaisseur du sol
    cols_n = int(math.ceil((max_x - min_x) / cell))
    rows_n = int(math.ceil((max_z - min_z) / cell))
    tiles = []
    for i in range(cols_n):
        for j in range(rows_n):
            cx = min_x + (i + 0.5) * cell
            cz = min_z + (j + 0.5) * cell
            t = np.array(
                [
                    [1, 0, 0, cx],
                    [0, 1, 0, min_y + h],
                    [0, 0, 1, cz],
                    [0, 0, 0, 1],
                ]
            )
            tile = trimesh.creation.box(
                extents=(cell, h, cell), transform=t
            )  # Alternance sombre/clair discrète
            if (i + j) % 2 == 0:
                color = [0.82, 0.87, 0.82, 1.0]
            else:
                color = [0.88, 0.93, 0.88, 1.0]
            tile.visual.vertex_colors = [color] * len(tile.vertices)
            tiles.append(tile)
    ground = trimesh.util.concatenate(tiles)
    scene.add(pyrender.Mesh.from_trimesh(ground, smooth=False))


# ─────────────────────────────────────────────
# Multi-scene loader  ← NOUVELLE FONCTION CLEF
# ─────────────────────────────────────────────


def load_all_scenes_from_folder(
    folder,
    model_folder="data/smplx_models",
    model_type="smplx",
    gender="NEUTRAL_2020",
    margin=2.0,
):
    npz_files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    print(f"Found {len(npz_files)} scenes → single line")

    loaded_scenes = []
    for npz_file in npz_files:
        try:
            result = load_smplx_data(
                npz_file, model_folder, model_type, gender, 10, 50, "cuda"
            )
            (
                vertices_dict,
                faces,
                _,
                _,
                emotions_dict,
                joints_dict,
                start_dict,
                stop_dict,
            ) = result

            all_vertices_all_frames = np.concatenate(
                [v.reshape(-1, 3) for v in vertices_dict.values()],
                axis=0,
            )

            cx = all_vertices_all_frames[:, 0].mean()
            cz = all_vertices_all_frames[:, 2].mean()
            min_y = all_vertices_all_frames[:, 1].min()  # sol GLOBAL

            center = np.array([0.0, 0.0, 0.0])
            # Emprise X sur toutes les frames
            all_frames = np.concatenate(
                [
                    (v - center[np.newaxis, :]).reshape(-1, 3)
                    for v in vertices_dict.values()
                ],
                axis=0,
            )
            extent_x = all_frames[:, 0].max() - all_frames[:, 0].min()

            loaded_scenes.append(
                {
                    "npz_file": npz_file,
                    "vertices_dict": vertices_dict,
                    "faces": faces,
                    "emotions_dict": emotions_dict,
                    "joints_dict": joints_dict,
                    "start_dict": start_dict,
                    "stop_dict": stop_dict,
                    "center": center,
                    "extent_x": extent_x,
                    "min_y": min_y,
                }
            )
        except Exception as e:
            print(f"error {os.path.basename(npz_file)}: {e}")

    # Offsets cumulatifs sur X uniquement
    cursor_x = 0.0
    for sc in loaded_scenes:
        sc["offset_x"] = cursor_x
        cursor_x += sc["extent_x"] + margin

    all_vertices = {}
    all_faces = None
    all_joints = {}
    all_emotions = {}
    all_starts = {}
    all_stops = {}
    scene_labels = {}

    for scene_idx, sc in enumerate(loaded_scenes):
        offset = np.array([sc["offset_x"], -sc["min_y"], 0.0])
        center = sc["center"]
        if all_faces is None:
            all_faces = sc["faces"]
        scene_name = os.path.splitext(os.path.basename(sc["npz_file"]))[0][:18]

        for key, verts_seq in sc["vertices_dict"].items():
            new_key = f"s{scene_idx:03d}__{key}"
            recentered_v = verts_seq - center[np.newaxis, np.newaxis, :]
            recentered_j = sc["joints_dict"][key] - center[np.newaxis, np.newaxis, :]
            all_vertices[new_key] = recentered_v + offset[np.newaxis, np.newaxis, :]
            all_joints[new_key] = recentered_j + offset[np.newaxis, np.newaxis, :]
            all_emotions[new_key] = sc["emotions_dict"].get(key)
            all_starts[new_key] = sc["start_dict"].get(key, 0)
            all_stops[new_key] = sc["stop_dict"].get(key, len(verts_seq))
            scene_labels[new_key] = scene_name

    return (
        all_vertices,
        all_faces,
        all_joints,
        all_emotions,
        all_starts,
        all_stops,
        scene_labels,
    )


# ─────────────────────────────────────────────
# MAIN GRID RENDER FUNCTION
# ─────────────────────────────────────────────


def render_folder_grid(
    folder,
    output_dir,
    output_file="grid_render.mp4",
    model_folder="data/smplx_models",
    resolution=(1920, 1080),
    fps=30,
    loop=True,  # loop shorter sequences to match the longest
    show_labels=True,  # overlay scene names
    camera_slowdown=1.0,
):
    """
    Render all .npz files from `folder` in a 3D grid layout, all at once.

    Parameters
    ----------
    folder               : path to directory of .npz files
    output_dir           : where to save the mp4
    output_file          : output filename
    model_folder         : SMPL-X model folder
    spacing              : (dx, dy, dz) between scene origins on the grid
    resolution           : (width, height) in pixels
    fps                  : frames per second
    loop                 : loop shorter sequences so all run for max_frames
    show_labels          : draw scene name above each group
    camera_elevation     : 0→horizontal  1→top-down  (blended)
    camera_distance_factor: multiply auto-computed distance
    """
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # ── Load all scenes ──────────────────────────────────────────────────
    (
        vertices_dict,
        faces,
        joints_dict,
        emotions_dict,
        start_dict,
        stop_dict,
        scene_labels,
    ) = load_all_scenes_from_folder(folder, model_folder)

    keys = sorted(vertices_dict.keys())

    PALETTE = [
        (0, 127, 255),
        (255, 127, 0),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 0, 127),
        (0, 255, 127),
        (127, 0, 255),
        (127, 255, 0),
    ]
    scene_color_map = {}
    for key in keys:
        scene_idx = int(key.split("__")[0][1:])  # extrait le numéro depuis "s003__..."
        scene_color_map[key] = PALETTE[scene_idx % len(PALETTE)]
    person_colors = scene_color_map
    os.makedirs(output_dir, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(*resolution)
    writer = cv2.VideoWriter(
        os.path.join(output_dir, output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        resolution,
    )

    # ── Frame count ──────────────────────────────────────────────────────
    seq_lengths = {k: stop_dict[k] - start_dict[k] for k in keys}
    max_frames = max(seq_lengths.values())

    # ── Camera setup ─────────────────────────────────────────────────────
    all_v0 = np.concatenate([v[0] for v in vertices_dict.values()], axis=0)
    total_x_min = all_v0[:, 0].min()
    total_x_max = all_v0[:, 0].max()
    avg_height = all_v0[:, 1].max() - all_v0[:, 1].min()

    cam_y = 1.5  # légèrement au-dessus des têtes
    cam_z = -3.0  # recul latéral fixe
    cam_x_start = total_x_min  # début de la ligne
    cam_x_end = total_x_max  # milieu = destination finale

    aspect_ratio = resolution[0] / resolution[1]
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=aspect_ratio)
    projection_matrix = cam.get_projection_matrix(aspect_ratio)

    # ── Lights ───────────────────────────────────────────────────────────
    light1 = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    lp1 = look_at(0.0 + np.array([10, 15, 10]), 0.0)
    light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    lp2 = look_at(0.0 + np.array([-10, 10, -5]), 0.0)
    light3 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    lp3 = look_at(0.0 + np.array([0, 12, -10]), 0.0)

    # ── Pre-build every mesh ──────────────────────────────────────────────
    print("Pre-building meshes (threaded)…")

    def _build_mesh(key, abs_frame):
        adjusted = abs_frame - start_dict[key]
        if adjusted < 0 or adjusted >= len(vertices_dict[key]):
            return key, abs_frame, None
        verts = vertices_dict[key][adjusted]
        c = person_colors[key]
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[c[0] / 255, c[1] / 255, c[2] / 255, 1.0],
            metallicFactor=0.15,
            roughnessFactor=0.55,
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return (
            key,
            abs_frame,
            pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True),
        )

    tasks = [(k, start_dict[k] + fi) for k in keys for fi in range(seq_lengths[k])]
    raw = Parallel(n_jobs=-1, backend="threading")(
        delayed(_build_mesh)(k, af) for k, af in tasks
    )
    mesh_cache = {}
    for key, af, mesh in raw:
        if mesh is not None:
            mesh_cache.setdefault(key, {})[af] = mesh

    print(f"  → {sum(len(v) for v in mesh_cache.values())} meshes ready")

    # ── Render loop ───────────────────────────────────────────────────────
    for frame_idx in tqdm(range(max_frames), desc="Rendering grid"):
        cam_duration = int(max_frames * camera_slowdown)
        t = min(frame_idx / max(cam_duration - 1, 1), 1.0)
        cam_x = cam_x_start + t * (cam_x_end - cam_x_start)

        camera_pos = np.array([cam_x, cam_y, cam_z])
        target = np.array([cam_x, avg_height * 0.25, 0.0])
        camera_pose = look_at(camera_pos, target)
        view_matrix = np.linalg.inv(camera_pose)
        scene = pyrender.Scene(
            bg_color=[0.8, 0.9, 1.0, 1.0],
            ambient_light=[0.25, 0.25, 0.25],
        )
        add_simple_ground(scene, vertices_dict)

        # Grouper par scène
        scenes_dict = {}
        for key in keys:
            scene_idx = int(key.split("__")[0][1:])  # extrait "s003" -> 3
            if scene_idx not in scenes_dict:
                scenes_dict[scene_idx] = []
            scenes_dict[scene_idx].append(key)

        for scene_idx, scene_keys in scenes_dict.items():
            # Durée max de la scène = max des durées de tous les personnages
            scene_max_len = max(stop_dict[k] - start_dict[k] for k in scene_keys)

            # Timer de la scène (loop par scène)
            if loop:
                scene_time = frame_idx % scene_max_len
            else:
                scene_time = frame_idx
                if scene_time >= scene_max_len:
                    continue

            # Pour chaque personnage de cette scène
            for key in scene_keys:
                start = start_dict[key]
                stop = stop_dict[key]
                person_len = stop - start

                # Si le personnage n'est pas encore actif à ce moment de la scène
                if scene_time >= person_len:
                    continue

                effective_frame = start + scene_time

                if effective_frame in mesh_cache.get(key, {}):
                    scene.add(mesh_cache[key][effective_frame])

        scene.add(cam, pose=camera_pose)
        scene.add(light1, pose=lp1)
        scene.add(light2, pose=lp2)
        scene.add(light3, pose=lp3)

        color, _ = renderer.render(
            scene,
            flags=(
                pyrender.RenderFlags.ALL_SOLID
                | pyrender.RenderFlags.SKIP_CULL_FACES
                | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
                | pyrender.RenderFlags.RGBA
            ),
        )
        bgr = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)

        # ── Scene label overlays ─────────────────────────────────────────
        if show_labels:
            drawn_labels = set()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1

            for scene_idx, scene_keys in scenes_dict.items():
                scene_max_len = max(stop_dict[k] - start_dict[k] for k in scene_keys)

                if loop:
                    scene_time = frame_idx % scene_max_len
                else:
                    scene_time = frame_idx
                    if scene_time >= scene_max_len:
                        continue

                for key in scene_keys:
                    start = start_dict[key]
                    stop = stop_dict[key]
                    person_len = stop - start

                    if scene_time >= person_len:
                        continue

                    label = scene_labels.get(key, "")
                    if label in drawn_labels:
                        continue

                    effective_frame = start + scene_time

                    if effective_frame not in mesh_cache.get(key, {}):
                        continue

                    jseq = joints_dict[key]
                    adj = scene_time
                    if adj >= len(jseq):
                        continue
                    joint_3d = (
                        jseq[adj][15]
                        if jseq.shape[1] > 15
                        else jseq[adj][np.argmax(jseq[adj, :, 1])]
                    )
                    label_3d = joint_3d + np.array([0.0, 0.4, 0.0])
                    proj = project_point_pyrender(
                        label_3d,
                        view_matrix,
                        projection_matrix,
                        resolution[0],
                        resolution[1],
                    )
                    if proj is None:
                        continue
                    px, py = proj
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    overlay = bgr.copy()
                    cv2.rectangle(
                        overlay,
                        (px - 4, py - th - 4),
                        (px + tw + 4, py + 4),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0, bgr)
                    cv2.putText(
                        bgr,
                        label,
                        (px, py),
                        font,
                        font_scale,
                        (220, 240, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
                    drawn_labels.add(label)

        writer.write(bgr.astype(np.uint8))

    writer.release()
    renderer.delete()
    out_path = os.path.join(output_dir, output_file)
    print(f"\n✓ Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# Original single-scene functions (unchanged)
# ─────────────────────────────────────────────


def render_multi_person_with_overlay(
    npz_file,
    output_dir,
    model_folder="data/smplx_models",
    model_type="smplx",
    gender="NEUTRAL_2020",
    output_file="test.mp4",
    emotion=False,
    follow_0=False,
):
    def prepare_renderer(resolution):
        return pyrender.OffscreenRenderer(*resolution)

    def _build_meshes_for_person(key):
        color = person_colors[key]
        verts_seq = vertices_dict[key]
        meshes = []
        for frame_idx in range(num_frames):
            adjusted_idx = frame_idx - frame_offsets[key]
            if adjusted_idx < 0 or adjusted_idx >= len(verts_seq):
                continue
            verts = verts_seq[adjusted_idx]
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[color[0] / 255, color[1] / 255, color[2] / 255, 1.0],
                metallicFactor=0.15,
                roughnessFactor=0.55,
                alphaMode="BLEND",
            )
            material.depthWrite = False
            material.depthTest = False
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            meshes.append(
                pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
            )
        return key, meshes

    os.environ["PYOPENGL_PLATFORM"] = "egl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices_dict, faces, _, _, emotions_dict, joints_dict, start_dict, stop_dict = (
        load_smplx_data(npz_file, model_folder, model_type, gender, 10, 50, device)
    )

    PALETTE = [
        (0, 127, 255),
        (255, 127, 0),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 0, 127),
        (0, 255, 127),
        (127, 0, 255),
        (127, 255, 0),
    ]
    keys = list(vertices_dict.keys())
    sorted_keys = sorted(keys, key=lambda x: int(x.split("body_")[-1]))
    keys = sorted_keys
    person_colors = {key: PALETTE[i % len(PALETTE)] for i, key in enumerate(keys)}

    os.makedirs(output_dir, exist_ok=True)
    resolution = (1080, 1080)
    renderer = prepare_renderer(resolution)
    writer = cv2.VideoWriter(
        os.path.join(output_dir, output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        resolution,
    )

    frame_lengths = {key: len(verts) for key, verts in vertices_dict.items()}
    max_frames = max(frame_lengths.values())
    frame_offsets = {key: max_frames - length for key, length in frame_lengths.items()}
    num_frames = max_frames

    center = np.array([0.01, 0.5, 0.01])
    aspect_ratio = resolution[0] / resolution[1]
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=aspect_ratio)
    projection_matrix = cam.get_projection_matrix(aspect_ratio)

    if follow_0 is False:
        camera_pos = np.array([0.01, 1.5, -3.01])
        camera_pose = look_at(camera_pos, center)
        view_matrix = np.linalg.inv(camera_pose)

    light = pyrender.PointLight(color=np.ones(3), intensity=5.0)
    lp1 = look_at(
        np.array([camera_pos[0] + 2, 3, camera_pos[2] + 2]),
        np.array([camera_pos[0], 1, camera_pos[2]]),
    )
    light2 = pyrender.PointLight(color=np.ones(3), intensity=15.0)
    lp2 = look_at(
        np.array([camera_pos[0] - 2, 5, camera_pos[2] + 3]),
        np.array([camera_pos[0], 1, camera_pos[2]]),
    )
    light3 = pyrender.PointLight(color=np.ones(3), intensity=7.0)
    lp3 = look_at(
        np.array([camera_pos[0], 3, camera_pos[2] - 4]),
        np.array([camera_pos[0], 1, camera_pos[2]]),
    )
    light4 = pyrender.DirectionalLight(color=np.ones(3), intensity=0.5)
    lp4 = look_at(
        np.array([camera_pos[0] + 2.5, 6, camera_pos[2] + 2.5]),
        np.array([camera_pos[0], 0.8, camera_pos[2]]),
    )

    person_meshes = Parallel(n_jobs=-1, backend="threading")(
        delayed(_build_meshes_for_person)(k) for k in vertices_dict.keys()
    )
    person_meshes = dict(person_meshes)

    for frame_idx in tqdm(range(num_frames), desc="Rendering video"):
        if follow_0:
            p0c = vertices_dict[list(person_meshes.keys())[0]][frame_idx].mean(axis=0)
            camera_pos = np.array(p0c + np.array([1e-8, 1e-8, -3.0]))
            camera_pose = look_at(camera_pos, p0c)
            view_matrix = np.linalg.inv(camera_pose)

        scene = pyrender.Scene(bg_color=[0.8, 0.9, 1.0], ambient_light=[0.2, 0.2, 0.2])
        add_simple_ground(scene, vertices_dict)

        for key in vertices_dict.keys():
            if start_dict[key] is None or stop_dict[key] is None:
                start_dict[key] = 0
                stop_dict[key] = num_frames
            if start_dict[key] <= frame_idx < stop_dict[key]:
                scene.add(person_meshes[key][frame_idx - start_dict[key]])

        scene.add(cam, pose=camera_pose)
        scene.add(light, pose=lp1)
        scene.add(light2, pose=lp2)
        scene.add(light3, pose=lp3)
        scene.add(light4, pose=lp4)

        color, _ = renderer.render(
            scene,
            flags=(
                pyrender.RenderFlags.ALL_SOLID
                | pyrender.RenderFlags.SKIP_CULL_FACES
                | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
                | pyrender.RenderFlags.RGBA
            ),
        )
        bgr_frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        if emotion:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for key, joints_seq in joints_dict.items():
                if key not in emotions_dict:
                    continue
                emotion_values = emotions_dict[key]
                if frame_idx <= start_dict[key] or frame_idx >= stop_dict[key]:
                    continue
                joints = joints_seq[frame_idx - start_dict[key]]
                joint_3d = (
                    joints[15]
                    if joints.shape[0] > 15
                    else joints[np.argmax(joints[:, 1])]
                )
                projected = project_point_pyrender(
                    joint_3d,
                    view_matrix,
                    projection_matrix,
                    resolution[0],
                    resolution[1],
                )
                if projected:
                    px, py = projected
                    emotion_str = (
                        "".join(emotion_values[frame_idx - start_dict[key]])
                        if isinstance(
                            emotion_values[frame_idx - start_dict[key]][0], str
                        )
                        else "emotion"
                    )
                    (tw, th), _ = cv2.getTextSize(emotion_str, font, 1, 2)
                    overlay = bgr_frame.copy()
                    tc = person_colors[key]
                    cv2.rectangle(
                        overlay,
                        (px - 50, py - th - 60),
                        (px + tw - 50, py - 50),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        overlay,
                        emotion_str,
                        (px - 50, py - 60),
                        font,
                        1,
                        (tc[2], tc[1], tc[0]),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.addWeighted(overlay, 0.5, bgr_frame, 0.5, 0, bgr_frame)

        writer.write(bgr_frame.astype(np.uint8))

    writer.release()
    print(f"Video saved to: {os.path.join(output_dir, output_file)}")
