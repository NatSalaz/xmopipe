import os
import glob
import cv2
import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c, colors
import trimesh
import pyrender
import smplx
import random
import torch
import joblib, multiprocessing, math
from joblib import Parallel, delayed  #  pip install joblib


def project_point_pyrender(point_3d, view_matrix, projection_matrix, width, height):
    point = np.append(point_3d, 1.0)
    clip = projection_matrix @ (view_matrix @ point)

    if clip[3] == 0:
        return None

    ndc = clip[:3] / clip[3]  # [-1,1]
    x_ndc, y_ndc = ndc[0], ndc[1]
    px = int((x_ndc + 1) * 0.5 * width)
    py = int((1 - y_ndc) * 0.5 * height)

    return px, py


def load_smplx_data(
    npz_file, model_folder, model_type, gender, num_betas, num_emotion_coeffs, device
):
    """Loads smplx data from npzs and get useful infos for scener rendering"""
    models = {}
    faces = None
    trans_dict = {}
    expressions_dict = {}
    emotions_dict = {}
    joints_dict = {}
    start_dict = {}
    stop_dict = {}
    scene_data = np.load(npz_file, allow_pickle=True)
    print(f"Loaded npz file: {npz_file}")
    print("Preparing rendering...")
    for person_id in scene_data.files:

        person_data = scene_data[person_id].item()
        start = person_data["start"] if "start" in person_data else None
        stop = person_data["stop"] if "stop" in person_data else None

        required_keys = ["poses", "betas", "trans", "cam_transl"]
        missing_keys = [k for k in required_keys if k not in person_data]
        if missing_keys:
            print(
                f"Warning: Incomplete data in {npz_file}, missing keys: {missing_keys}, file skipped"
            )
            continue

        if "expressions" not in person_data and "emotions" not in person_data:
            print(
                f"Warning: Neither expressions nor emotions found in {npz_file}, file skipped"
            )
            continue

        required_keys = ["poses", "betas", "trans", "cam_transl"]
        missing_keys = [k for k in required_keys if k not in person_data]
        if missing_keys:
            print(
                f"Warning: Incomplete data in {npz_file} → {person_id}, missing keys: {missing_keys}"
            )
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

        expr = person_data["expressions"] if "expressions" in person_data else None
        emot = person_data["emotions"] if "emotions" in person_data else None

        expr_tensor = torch.tensor(expr).float().cuda()
        expressions_dict[f"{os.path.basename(npz_file)}__{person_id}"] = expr
        emotions_dict[f"{os.path.basename(npz_file)}__{person_id}"] = (
            emot
            if emot is not None
            else np.array([[f"{v:.2f}" for v in frame] for frame in expr])
        )
        model = smplx.create(
            model_folder,
            model_type=model_type,
            gender=gender,
            num_betas=num_betas,
            num_expression_coeffs=num_emotion_coeffs,
            use_pca=False,
        ).cuda()
        # print("root:", torch.tensor(poses[:, :3]).float().cuda()[:10])
        # "trans:", trans[:10] - trans[0])
        output = model(
            betas=torch.tensor(betas).float().cuda(),
            transl=torch.tensor(trans).float().cuda(),
            expression=expr_tensor,
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
    if not models:
        raise ValueError("No models loaded. Please check your .npz files.")

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


def add_simple_ground(scene, vertices_dict):
    all_vertices = []

    for verts_seq in vertices_dict.values():
        if len(verts_seq) == 0:
            continue
        all_vertices.append(verts_seq[0])

    all_vertices = np.concatenate(all_vertices, axis=0)

    min_x, max_x = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    min_z, max_z = all_vertices[:, 2].min(), all_vertices[:, 2].max()
    min_x, max_x = -5.0, 5.0
    min_z, max_z = -5.0, 5.0
    min_y = all_vertices[:, 1].min()

    center_x = (min_x + max_x) / 2
    center_z = (min_z + max_z) / 2
    size_x = max_x - min_x
    size_z = max_z - min_z

    h = 0.001
    transform = np.array(
        [
            [1, 0, 0, center_x],
            [0, 1, 0, min_y + h],
            [0, 0, 1, center_z],
            [0, 0, 0, 1],
        ]
    )

    box = trimesh.creation.box(extents=(size_x, h, size_z), transform=transform)
    box.visual.vertex_colors = [(0.9, 0.95, 0.9, 1.0)] * len(box.vertices)

    scene.add(pyrender.Mesh.from_trimesh(box, smooth=False))


def render_multi_person_with_overlay(
    npz_file,
    output_dir,
    model_folder="data/smplx_models",
    model_type="smplx",
    gender="NEUTRAL_2020",
    output_file="test.mp4",
    emotion=False,
    follow_0=False,
    from_above=False,
    text_overlay=None,
    skeleton=False,
):
    SKELETON_LINKS = [
        (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
        (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
        (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
        (7, 60), (8, 63),
        (76, 77), (77, 78), (78, 79), (79, 80),
        (81, 82), (82, 83), (83, 84), (84, 85),
    ]

    PALETTE = [
        (85, 170, 255), (255, 130, 85), (255, 85, 85), (85, 255, 85),
        (255, 255, 85), (85, 85, 255), (85, 255, 255), (255, 85, 255),
        (255, 85, 170), (85, 255, 170), (170, 85, 255), (170, 255, 85),
    ]

    def _build_meshes_for_person(key):
        color = person_colors[key]
        verts_seq = vertices_dict[key]
        start = start_dict[key]
        meshes = {}
        for adjusted, verts in enumerate(verts_seq):
            frame_idx = start + adjusted
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[color[0] / 255, color[1] / 255, color[2] / 255, 1.0],
                metallicFactor=0.15, roughnessFactor=0.55, alphaMode="BLEND",
            )
            material.depthWrite = False
            material.depthTest = False
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            meshes[frame_idx] = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        return key, meshes

    def _build_skeleton_for_frame(args):
        key, frame_idx, joints, color = args
        color_full = list(color) + [255]
        linked = set(i for link in SKELETON_LINKS for i in link)
        parts = []

        face_indices = set(range(22, 25)) | set(range(86, len(joints)))
        for idx in linked | face_indices:
            if idx >= len(joints):
                continue
            radius = 0.008 if idx in face_indices else 0.015
            sph = trimesh.creation.icosphere(subdivisions=1, radius=radius)
            sph.visual.vertex_colors = [color_full] * len(sph.vertices)
            sph.apply_translation(joints[idx])
            parts.append(sph)

        for i, j in SKELETON_LINKS:
            if i >= len(joints) or j >= len(joints):
                continue
            pt1, pt2 = joints[i], joints[j]
            vec = pt2 - pt1
            length = np.linalg.norm(vec)
            if length < 1e-6:
                continue
            direction = vec / length
            radius = 0.005 if max(i, j) >= 21 else 0.01
            cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=4)
            cyl.visual.vertex_colors = [color_full] * len(cyl.vertices)
            z_axis = np.array([0, 0, 1])
            axis = np.cross(z_axis, direction)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            if np.linalg.norm(axis) > 1e-6:
                rot = trimesh.transformations.rotation_matrix(angle, axis)
                cyl.apply_transform(rot)
            cyl.apply_translation((pt1 + pt2) / 2)
            parts.append(cyl)

        if parts:
            return (key, frame_idx, trimesh.util.concatenate(parts))
        return (key, frame_idx, None)

    os.environ["PYOPENGL_PLATFORM"] = "egl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices_dict, faces, _, _, emotions_dict, joints_dict, start_dict, stop_dict = (
        load_smplx_data(npz_file, model_folder, model_type, gender, 10, 50, device)
    )

    keys = sorted(vertices_dict.keys(), key=lambda x: int(x.split("body_")[-1]))
    person_colors = {key: PALETTE[i % len(PALETTE)] for i, key in enumerate(keys)}

    # Normalize None starts/stops
    num_frames = max(len(j) for j in joints_dict.values())
    for key in keys:
        if start_dict[key] is None:
            start_dict[key] = 0
        if stop_dict[key] is None:
            stop_dict[key] = num_frames

    os.makedirs(output_dir, exist_ok=True)
    resolution = (1080, 1080)
    renderer = pyrender.OffscreenRenderer(*resolution)
    writer = cv2.VideoWriter(
        os.path.join(output_dir, output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30, resolution,
    )

    # Camera (identical for both modes)
    center = np.mean([joints_dict[key][0][0] for key in keys], axis=0)
    aspect_ratio = resolution[0] / resolution[1]
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=aspect_ratio)
    projection_matrix = cam.get_projection_matrix(aspect_ratio)

    camera_pos = np.array([-0.01, 1.0, -5.01])
    if not follow_0:
        if from_above:
            camera_pos = np.array([-0.01, 5.0, -5.01])
        camera_pose = look_at(camera_pos + center, center)
        view_matrix = np.linalg.inv(camera_pose)

    # Lights (PointLights for both modes)
    light = pyrender.PointLight(color=np.ones(3), intensity=10.0)
    light_pose = look_at(
        np.array([camera_pos[0] + 2, 3, camera_pos[2] + 2]),
        np.array([camera_pos[0], 1, camera_pos[2]]),
    )
    light2 = pyrender.PointLight(color=np.ones(3), intensity=30.0)
    light2_pose = look_at(
        np.array([camera_pos[0] - 2, 5, camera_pos[2] + 3]),
        np.array([camera_pos[0], 1, camera_pos[2]]),
    )
    light3 = pyrender.PointLight(color=np.ones(3), intensity=15.0)
    light3_pose = look_at(
        np.array([camera_pos[0], 3, camera_pos[2] - 4]),
        np.array([camera_pos[0], 1, camera_pos[2]]),
    )
    light4 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    light4_pose = look_at(
        np.array([camera_pos[0] + 2.5, 6, camera_pos[2] + 2.5]),
        np.array([camera_pos[0], 0.8, camera_pos[2]]),
    )

    # Tiled ground
    y_ground = np.min([j[0][:, 1].min() for j in joints_dict.values()])
    cubes = []
    for i in range(10):
        for j in range(10):
            transform = np.array([
                [1, 0, 0, -25 + i * 5],
                [0, 1, 0, y_ground],
                [0, 0, 1, -25 + j * 5],
                [0, 0, 0, 1],
            ])
            cube = trimesh.creation.box(extents=(5, 0.1, 5), transform=transform)
            c = (0.8, 0.9, 0.8) if (i + j) % 2 == 0 else (0.9, 1.0, 0.9)
            cube.visual.vertex_colors = [list(c) + [1.0]] * len(cube.vertices)
            cubes.append(cube)
    ground_mesh = pyrender.Mesh.from_trimesh(trimesh.util.concatenate(cubes), smooth=False)

    # Pre-build render cache
    if skeleton:
        print("Precalculating skeleton meshes...")
        tasks = [
            (key, frame_idx, joints_dict[key][frame_idx - start_dict[key]], person_colors[key])
            for key in keys
            for frame_idx in range(start_dict[key], stop_dict[key])
            if frame_idx - start_dict[key] < len(joints_dict[key])
        ]
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_build_skeleton_for_frame)(task) for task in tasks
        )
        render_cache = {}
        for key, frame_idx, tm in results:
            if tm is not None:
                render_cache.setdefault(key, {})[frame_idx] = pyrender.Mesh.from_trimesh(tm, smooth=True)
        print(f"{len(results)} precalculated meshes!")
    else:
        pairs = Parallel(n_jobs=-1, backend="threading")(
            delayed(_build_meshes_for_person)(k) for k in keys
        )
        render_cache = dict(pairs)

    # text_overlay pre-compute
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_params = None
    if text_overlay:
        thickness = 2
        margin, padding = 20, 10
        max_width = resolution[0] - 2 * margin - 2 * padding
        font_scale = 0.8
        (text_w, text_h), baseline = cv2.getTextSize(text_overlay, font, font_scale, thickness)
        while text_w > max_width and font_scale > 0.3:
            font_scale -= 0.05
            (text_w, text_h), baseline = cv2.getTextSize(text_overlay, font, font_scale, thickness)
        x = resolution[0] - text_w - margin
        y_txt = resolution[1] - margin
        text_params = {
            "text": text_overlay, "font": font, "font_scale": font_scale,
            "thickness": thickness, "text_color": (255, 255, 255),
            "rect_pt1": (x - padding, y_txt - text_h - padding),
            "rect_pt2": (x + text_w + padding, y_txt + baseline + padding),
            "bg_color": (0, 0, 0), "text_pos": (x, y_txt),
        }

    for frame_idx in tqdm(range(num_frames), desc="Rendering"):
        if follow_0:
            p0_center = joints_dict[keys[0]][frame_idx].mean(axis=0)
            camera_pos = p0_center + np.array([1e-8, 1e-8, -3.0])
            camera_pose = look_at(camera_pos, p0_center)
            view_matrix = np.linalg.inv(camera_pose)

        scene = pyrender.Scene(bg_color=[0.8, 0.9, 1.0], ambient_light=[0.2, 0.2, 0.2])
        scene.add(ground_mesh)

        for key in keys:
            if not (start_dict[key] <= frame_idx < stop_dict[key]):
                continue
            if key in render_cache and frame_idx in render_cache[key]:
                scene.add(render_cache[key][frame_idx])

        scene.add(cam, pose=camera_pose)
        scene.add(light, pose=light_pose)
        scene.add(light2, pose=light2_pose)
        scene.add(light3, pose=light3_pose)
        scene.add(light4, pose=light4_pose)

        color_img, _ = renderer.render(
            scene,
            flags=pyrender.RenderFlags.ALL_SOLID
            | pyrender.RenderFlags.SKIP_CULL_FACES
            | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            | pyrender.RenderFlags.RGBA,
        )
        bgr_frame = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        if emotion:
            for key in keys:
                if key not in emotions_dict:
                    continue
                start = start_dict[key]
                stop = stop_dict[key]
                if not (start <= frame_idx < stop):
                    continue
                adjusted = frame_idx - start
                if adjusted >= len(emotions_dict[key]):
                    continue
                joints = joints_dict[key][adjusted]
                joint_3d = joints[15] if len(joints) > 15 else joints[np.argmax(joints[:, 1])]
                projected = project_point_pyrender(
                    joint_3d, view_matrix, projection_matrix, resolution[0], resolution[1]
                )
                if projected:
                    px, py = projected
                    emotion_str = (
                        "".join(emotions_dict[key][adjusted])
                        if isinstance(emotions_dict[key][adjusted][0], str)
                        else "emotion"
                    )
                    (text_w, text_h), _ = cv2.getTextSize(emotion_str, font, 1, 2)
                    overlay = bgr_frame.copy()
                    text_color = person_colors[key]
                    cv2.rectangle(overlay, (px-50, py-text_h-60), (px+text_w-50, py-50), (0, 0, 0), -1)
                    cv2.putText(overlay, emotion_str, (px-50, py-60), font, 1,
                                (text_color[2], text_color[1], text_color[0]), 2, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.5, bgr_frame, 0.5, 0, bgr_frame)

        if text_params:
            cv2.rectangle(bgr_frame, text_params["rect_pt1"], text_params["rect_pt2"], text_params["bg_color"], -1)
            cv2.putText(bgr_frame, text_params["text"], text_params["text_pos"],
                        text_params["font"], text_params["font_scale"],
                        text_params["text_color"], text_params["thickness"], cv2.LINE_AA)

        writer.write(bgr_frame.astype(np.uint8))

    writer.release()
    renderer.delete()
    print(f"Video saved to: {os.path.join(output_dir, output_file)}")


def render_multi_person_with_overlay_skeleton(
    npz_file,
    output_dir,
    model_folder="data/smplx_models",
    model_type="smplx",
    gender="NEUTRAL_2020",
    output_file="output_joints.mp4",
    emotion=True,
    follow_0=False,
    from_above=False,
    text_overlay=None,
):
    return render_multi_person_with_overlay(
        npz_file=npz_file, output_dir=output_dir, model_folder=model_folder,
        model_type=model_type, gender=gender, output_file=output_file,
        emotion=emotion, follow_0=follow_0, from_above=from_above,
        text_overlay=text_overlay, skeleton=True,
    )




def render_single_frame_mesh(
    npz_file,
    output_path,
    frame_idx,
    model_folder="data/smplx_models",
    model_type="smplx",
    gender="NEUTRAL_2020",
    follow_0=False,
    resolution=(4096, 4096),
    body=None,
):
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices_dict, faces, _, _, _, _, start_dict, stop_dict = load_smplx_data(
        npz_file, model_folder, model_type, gender, 10, 50, device
    )
    print(vertices_dict.keys())
    keys = sorted(vertices_dict.keys(), key=lambda x: int(x.split("body_")[-1]))

    PALETTE = [
        (85, 170, 255),
        (255, 130, 85),
        (255, 85, 85),
        (85, 255, 85),
        (255, 255, 85),
        (85, 85, 255),
    ]
    person_colors = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}

    renderer = pyrender.OffscreenRenderer(*resolution)

    # Camera
    center = np.array([0.0, 0.9, 0.0])
    aspect_ratio = resolution[0] / resolution[1]
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=aspect_ratio)

    if follow_0 and keys:
        verts0 = vertices_dict[keys[0]][frame_idx - start_dict.get(keys[0], 0)]
        center = verts0.mean(axis=0)

    camera_pos = center + np.array([0.0, 0.8, -4.0])
    camera_pose = look_at(camera_pos, center)

    # Lights
    lights = [
        (
            pyrender.DirectionalLight(color=np.ones(3), intensity=2.5),
            look_at(np.array([5, 6, 5]), center),
        ),
        (
            pyrender.DirectionalLight(color=np.ones(3), intensity=1.5),
            look_at(np.array([-5, 5, -5]), center),
        ),
    ]

    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],  # TRANSPARENT
        ambient_light=[0.25, 0.25, 0.25],
    )

    # Meshes
    for key in keys:
        print(key)
        if body is None or f"body_{body}" in key:
            start = start_dict.get(key, 0)
            stop = stop_dict.get(key, 1e9)
            if not (start <= frame_idx < stop):
                continue

            verts_seq = vertices_dict[key]
            verts = verts_seq[frame_idx - start]

            color = person_colors[key]
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    color[0] / 255,
                    color[1] / 255,
                    color[2] / 255,
                    1.0,
                ],
                metallicFactor=0.15,
                roughnessFactor=0.55,
                alphaMode="BLEND",
            )
            material.depthWrite = False
            material.depthTest = False

            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))

    scene.add(cam, pose=camera_pose)
    for light, pose in lights:
        scene.add(light, pose=pose)

    # Render
    color_rgba, _ = renderer.render(
        scene,
        flags=(
            pyrender.RenderFlags.ALL_SOLID
            | pyrender.RenderFlags.SKIP_CULL_FACES
            | pyrender.RenderFlags.RGBA
        ),
    )

    renderer.delete()

    # Save PNG RGBA
    cv2.imwrite(
        output_path,
        cv2.cvtColor(color_rgba, cv2.COLOR_RGBA2BGRA),
    )

    print(f"Saved mesh frame {frame_idx} ==> {output_path}")
