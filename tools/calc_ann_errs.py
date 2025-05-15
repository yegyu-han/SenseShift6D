"""
Place this script under gdrnpp_bop2022/core/gdrn_modeling/tools/ss6d
"""
import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../../../..")
sys.path.insert(0, PROJ_ROOT)

from lib.vis_utils.colormap import colormap
from lib.utils.mask_utils import mask2bbox_xyxy, cocosegm2mask, get_edge
from core.utils.data_utils import read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.vis_utils.image import grid_show
from lib.vis_utils.image import vis_image_bboxes_cv2

glob_diff_list = []
glob_diff_abs_list = []

# Object info
id2obj = {1: "spray", 2:"pringles", 3:"tincase"}
objects = list(id2obj.values())

# Base settings
width = 1280
height = 720
depth_scale = 0.001  # mm to meter

# Model path settings
model_dir = "datasets/BOP_DATASETS/SenseShift6D/models/"
model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]

for scene_num in range(3): # object id. 0: spray, 1: pringles, 2: tincase
    obj_name = id2obj[scene_num +1]

    """
        TRAIN 
    """

    # Renderer initialization
    ren = EGLRenderer(
        model_paths,
        vertex_scale=depth_scale,
        use_cache=True,
        width=width,
        height=height,
    )

    # Dataset registration and loading
    dataset_name = f"ss6d_{obj_name}_lv3ae_tr"
    register_datasets([dataset_name])
    meta = MetadataCatalog.get(dataset_name)
    dset_dicts = DatasetCatalog.get(dataset_name)

    # tensor for rendering
    import torch
    tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
    # seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    pc_cam_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

    # Iterate through all images
    for d in tqdm(dset_dicts):
        K = d["cam"]
        scene_im_id = d["scene_im_id"]
        scene_id, im_id = scene_im_id.split("/")
        im_id = int(im_id)

        # Load GT pose
        annos = d["annotations"]
        cat_ids = [anno["category_id"] for anno in annos]
        obj_names = [meta.objs[cid] for cid in cat_ids]
        quats = [anno["quat"] for anno in annos]
        transes = [anno["trans"] for anno in annos]
        Rs = [quat2mat(q) for q in quats]
        gt_labels = [objects.index(name) for name in obj_names]
        gt_poses = [np.hstack([R, t.reshape(3, 1)]) for R, t in zip(Rs, transes)]

        # Rendered depth map
        # ren.render(gt_labels, gt_poses, K=K, seg_tensor=seg_tensor, pc_cam_tensor=pc_cam_tensor)
        ren.render(gt_labels, gt_poses, K=K, pc_cam_tensor=pc_cam_tensor)
        depth_render = pc_cam_tensor[:, :, 2].detach().cpu().numpy()

        # Capture depth map
        depth_path = osp.join(PROJ_ROOT, d["depth_file"])
        if not osp.exists(depth_path):
            print(f"[WARNING] Missing depth image: {depth_path}")
            continue
        depth_real = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_real *= depth_scale   # mm to meter

        # Compare only valid pixels     
        valid_mask = (depth_real > 0) & (depth_render > 0)
        if np.count_nonzero(valid_mask) == 0:
            continue

        diff = depth_real[valid_mask] - depth_render[valid_mask]
        diff_abs = np.abs(diff)

        # Remove outliers
        diff = diff[diff_abs < 0.05] 
        diff_abs = diff_abs[diff_abs < 0.05] 

        if len(diff) > 0:
            glob_diff_list.append(diff)
        if len(diff_abs) > 0:
            glob_diff_abs_list.append(diff_abs)


    """
        Test
    """

    ren = EGLRenderer(
        model_paths,
        vertex_scale=depth_scale,
        use_cache=True,
        width=width,
        height=height,
    )

    dataset_name = f"ss6d_{obj_name}_lv3ae_d1_te"
    register_datasets([dataset_name])
    meta = MetadataCatalog.get(dataset_name)
    dset_dicts = DatasetCatalog.get(dataset_name)

    import torch
    tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
    # seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    pc_cam_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

    for d in tqdm(dset_dicts):
        K = d["cam"]
        scene_im_id = d["scene_im_id"]
        scene_id, im_id = scene_im_id.split("/")
        im_id = int(im_id)

        annos = d["annotations"]
        cat_ids = [anno["category_id"] for anno in annos]
        obj_names = [meta.objs[cid] for cid in cat_ids]
        quats = [anno["quat"] for anno in annos]
        transes = [anno["trans"] for anno in annos]
        Rs = [quat2mat(q) for q in quats]
        gt_labels = [objects.index(name) for name in obj_names]
        gt_poses = [np.hstack([R, t.reshape(3, 1)]) for R, t in zip(Rs, transes)]

        # ren.render(gt_labels, gt_poses, K=K, seg_tensor=seg_tensor, pc_cam_tensor=pc_cam_tensor)
        ren.render(gt_labels, gt_poses, K=K, pc_cam_tensor=pc_cam_tensor)
        depth_render = pc_cam_tensor[:, :, 2].detach().cpu().numpy()

        depth_path = osp.join(PROJ_ROOT, d["depth_file"])
        if not osp.exists(depth_path):
            print(f"[WARNING] Missing depth image: {depth_path}")
            continue
        depth_real = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_real *= depth_scale

        valid_mask = (depth_real > 0) & (depth_render > 0)
        if np.count_nonzero(valid_mask) == 0:
            continue

        diff = depth_real[valid_mask] - depth_render[valid_mask]
        diff_abs = np.abs(diff)
        diff = diff[diff_abs < 0.05]  
        diff_abs = diff_abs[diff_abs < 0.05] 

        if len(diff) > 0:
            glob_diff_list.append(diff)
        if len(diff_abs) > 0:
            glob_diff_abs_list.append(diff_abs)


# ----------------- Total statistics ---------------------

if len(glob_diff_list) > 0:
    all_diffs = np.concatenate(glob_diff_list)
    mean_diff = np.mean(all_diffs)
    median_diff = np.median(all_diffs)
    std_diff = np.std(all_diffs)

    print(f"[TOTAL] GT pose accuracy statistics (diff)")
    print(f"Mean: {mean_diff * 1000:.2f} mm")
    print(f"Median: {median_diff * 1000:.2f} mm")
    print(f"Std: {std_diff * 1000:.2f} mm")
else:
    print("No valid comparisons found for diff.")

if len(glob_diff_abs_list) > 0:
    all_diffs = np.concatenate(glob_diff_abs_list)
    mean_diff = np.mean(all_diffs)
    median_diff = np.median(all_diffs)
    std_diff = np.std(all_diffs)

    print(f"[TOTAL] GT pose accuracy statistics (diff_abs)")
    print(f"Mean: {mean_diff * 1000:.2f} mm")
    print(f"Median: {median_diff * 1000:.2f} mm")
    print(f"Std: {std_diff * 1000:.2f} mm")
else:
    print("No valid comparisons found for diff_abs.")