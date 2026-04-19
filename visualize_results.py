import argparse
import os

import cv2
import numpy as np
import trimesh

from camera import Camera, cam_view2pose
import image
from transforms import transform_point3s

LIST_OBJ_FOLDERNAME = [
    "004_sugar_box",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "011_banana",
    "024_bowl",
]

PROJECTION_MODE = "opencv_K_camview"


def project_points(pts_world, view_matrix, intrinsic_matrix):
    """Project using cam_view2pose + K (same as depth / ICP chain)."""
    vm = np.asarray(view_matrix, dtype=np.float64)
    K = np.asarray(intrinsic_matrix, dtype=np.float64)
    cam_pose = cam_view2pose(vm)
    T_cw = np.linalg.inv(cam_pose)
    pts_cam = transform_point3s(T_cw, pts_world)
    z = pts_cam[:, 2]
    valid = np.abs(z) > 1e-8
    u = np.zeros(len(pts_world))
    v = np.zeros(len(pts_world))
    u[valid] = K[0, 0] * pts_cam[valid, 0] / z[valid] + K[0, 2]
    v[valid] = K[1, 1] * pts_cam[valid, 1] / z[valid] + K[1, 2]
    return np.stack([u, v], axis=1), valid


def build_triptych(dataset_dir, scene_id, my_camera, seed=0):
    """RGB | predicted mask | pose overlay for one scene."""
    np.random.seed(seed)
    rgb = image.read_rgb(f"{dataset_dir}rgb/{scene_id}_rgb.png")
    pred_mask = image.read_mask(f"{dataset_dir}pred/{scene_id}_pred.png")
    view_matrix = np.load(f"{dataset_dir}view_matrix/{scene_id}.npy")
    img_h, img_w = rgb.shape[0], rgb.shape[1]

    overlay = rgb.copy()
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    poses_loaded = 0
    for i in range(len(LIST_OBJ_FOLDERNAME)):
        pose_path = f"{dataset_dir}pred_pose/predmask/{scene_id}_{i+1}.npy"
        if os.path.exists(pose_path):
            poses_loaded += 1
            pose = np.load(pose_path)
            mesh_path = f"./YCB_subsubset/{LIST_OBJ_FOLDERNAME[i]}/model_com.obj"
            mesh = trimesh.load(mesh_path)
            pts_mesh = mesh.vertices
            if len(pts_mesh) > 500:
                indices = np.random.choice(len(pts_mesh), 500, replace=False)
                pts_mesh = pts_mesh[indices]
            pts_homo = np.hstack([pts_mesh, np.ones((len(pts_mesh), 1))])
            pts_world = (pose @ pts_homo.T).T[:, :3]
            pts_2d, valid = project_points(pts_world, view_matrix, my_camera.intrinsic_matrix)
            for row, pt in enumerate(pts_2d):
                if not valid[row]:
                    continue
                uu, vv = int(pt[0]), int(pt[1])
                if 0 <= uu < img_w and 0 <= vv < img_h:
                    cv2.circle(overlay, (uu, vv), 1, colors[i], -1)

    mask_viz = (pred_mask * 50).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask_viz, cv2.COLORMAP_JET)
    mask_color[pred_mask == 0] = 0
    combined = np.hstack([rgb, mask_color, overlay])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Input RGB", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Predicted Mask", (330, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(
        combined,
        f"Pose overlay ({PROJECTION_MODE}) scene={scene_id}",
        (580, 30),
        font,
        0.45,
        (255, 255, 255),
        2,
    )
    return combined, poses_loaded


def _discover_val_scenes(dataset_dir):
    """Scene ids that have rgb + view_matrix."""
    rgb_dir = os.path.join(dataset_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        return []
    ids = []
    for name in os.listdir(rgb_dir):
        if name.endswith("_rgb.png"):
            try:
                sid = int(name.replace("_rgb.png", ""))
                vm = os.path.join(dataset_dir, "view_matrix", f"{sid}.npy")
                if os.path.isfile(vm):
                    ids.append(sid)
            except ValueError:
                continue
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(description="RGB | mask | pose overlays for README")
    parser.add_argument(
        "--dataset",
        default="./dataset/val/",
        help="Dataset root (default: val)",
    )
    parser.add_argument(
        "--scenes",
        default="all",
        help='Comma-separated scene ids (e.g. 0,1,2) or "all"',
    )
    parser.add_argument(
        "--out",
        default="./assets/",
        help="Output directory",
    )
    parser.add_argument(
        "--montage",
        action="store_true",
        help="Also write readme_montage.png stacking all scenes vertically",
    )
    args = parser.parse_args()

    dataset_dir = os.path.normpath(args.dataset) + os.sep
    assets_dir = args.out
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    if args.scenes.strip().lower() == "all":
        scene_ids = _discover_val_scenes(dataset_dir)
        if not scene_ids:
            scene_ids = list(range(5))
    else:
        scene_ids = [int(x.strip()) for x in args.scenes.split(",") if x.strip()]

    my_camera = Camera(image_size=(240, 320), near=0.01, far=10.0, fov_width=69.40)

    panels = []
    for scene_id in scene_ids:
        combined, npose = build_triptych(dataset_dir, scene_id, my_camera)
        out_name = f"readme_candidate_scene{scene_id}.png"
        out_path = os.path.join(assets_dir, out_name)
        image.write_rgb(combined, out_path)
        print(f"Wrote {out_path} ({npose} poses)")
        panels.append(combined)

    primary = os.path.join(assets_dir, "visualization.png")
    if panels:
        image.write_rgb(panels[0], primary)
        print(f"Wrote {primary} (copy of scene {scene_ids[0]})")

    if args.montage and len(panels) > 1:
        montage = np.vstack(panels)
        mp = os.path.join(assets_dir, "readme_montage.png")
        image.write_rgb(montage, mp)
        print(f"Wrote {mp} ({len(panels)} scenes)")


if __name__ == "__main__":
    main()
