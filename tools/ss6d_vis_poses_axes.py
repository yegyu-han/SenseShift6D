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

score_thr = 0.3
colors = colormap(rgb=False, maximum=255)

# object info
id2obj = {1: "spray", 2: "pringles", 3: "tincase", 4: "sandwich", 5: "mouse"}
objects = list(id2obj.values())


def load_predicted_csv(fname):
    df = pd.read_csv(fname)
    if "score" not in df.columns:
        df["score"] = 1

    if "time" not in df.columns:
        df["time"] = -1

    info_list = df.to_dict("records")
    return info_list


def parse_Rt_in_csv(_item):
    if isinstance(_item, np.ndarray):
        return _item.astype(float)
    elif isinstance(_item, str):
        return np.array([float(i) for i in _item.strip().split()])
    else:
        raise TypeError(f"Unsupported type: {type(_item)}")

# Camera info
width = 1280
height = 720

tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

model_dir = "/SenseShift6D/models/"

model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]

ren = EGLRenderer(
    model_paths,
    vertex_scale=0.001,
    use_cache=True,
    width=width,
    height=height,
)

pred_path = "/gdrnpp_bop2022/core/gdrn_modeling/tools/ss6d/final_outputs/iclr/hipose/csv/SenseShift6D_tincase_B5_1_E2500G48.csv"
vis_dir = "/gdrnpp_bop2022/core/gdrn_modeling/tools/ss6d/final_outputs/iclr/hipose/tincase/" 
bbox_path = "/SenseShift6D/test/test_bboxes/scene_gt_info_bboxes.json"

mmcv.mkdir_or_exist(vis_dir)

preds_csv = load_predicted_csv(pred_path)
pred_bboxes = mmcv.load(bbox_path)
preds = {}
for item in preds_csv:
    im_key = "{}/{}".format(item["scene_id"], item["im_id"])
    item["time"] = float(item["time"])
    item["score"] = float(item["score"])
    item["R"] = parse_Rt_in_csv(item["R"]).reshape(3, 3)
    item["t"] = parse_Rt_in_csv(item["t"]) 
    item["t"] = parse_Rt_in_csv(item["t"]) / 1000
    # item["obj_name"] = id2obj[item["obj_id"]]

    if im_key not in preds:
        preds[im_key] = []
    preds[im_key].append(item)

dataset_name = "ss6d_{}_{}{}_d{}_te".format('tincase', 'lv1', 'E2500G48'.lower(),1)
register_datasets([dataset_name])

meta = MetadataCatalog.get(dataset_name)
print("MetadataCatalog: ", meta)
objs = meta.objs
dset_dicts = DatasetCatalog.get(dataset_name)
for d in tqdm(dset_dicts):
    K = d["cam"]
    file_name = d["file_name"]
    scene_im_id = d["scene_im_id"]

    img = read_image_mmcv(file_name, format="BGR")

    scene_im_id_split = d["scene_im_id"].split("/")
    scene_id = scene_im_id_split[0]
    im_id = int(scene_im_id_split[1])

    imH, imW = img.shape[:2]

    if scene_im_id not in preds:
        print(f"[{scene_im_id}] not detected in predictions. Skipping.")
        continue
    
    cur_preds = preds[scene_im_id]
    cur_bboxes = pred_bboxes[scene_im_id]
    
    est_Rs = []
    est_ts = []
    est_labels = []
    for pred_i, pred in enumerate(cur_preds):
        try:
            R_est = pred["R"]
            t_est = pred["t"]
            score = pred["score"]
            # obj_name = pred["obj_name"]
            obj_name = id2obj[item['obj_id']]
        except KeyError as e:
            print(f"Skipping prediction {pred_i} due to missing key: {e}")
            continue
        if score < score_thr:
            continue

        est_Rs.append(R_est)
        est_ts.append(t_est)
        est_labels.append(objects.index(obj_name))

    bboxes = []
    labels = []
    for _i, bbox in enumerate(cur_bboxes):
        score = bbox["score"]
        if score < score_thr:
            continue
        x1, y1, w1, h1 = bbox["bbox_est"]
        x2 = x1 + w1
        y2 = y1 + h1
        bboxes.append(np.array([x1, y1, x2, y2]))
        labels.append(str(bbox["obj_id"]))

    img_bbox = vis_image_bboxes_cv2(img, bboxes, labels)

    im_gray = mmcv.bgr2gray(img, keepdim=True)
    im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

    gt_Rs = []
    gt_ts = []
    gt_labels = []

    annos = d["annotations"]
    cat_ids = [anno["category_id"] for anno in annos]
    obj_names = [objs[cat_id] for cat_id in cat_ids]

    quats = [anno["quat"] for anno in annos]
    transes = [anno["trans"] for anno in annos]
    Rs = [quat2mat(quat) for quat in quats]
    for anno_i, anno in enumerate(annos):
        obj_name = obj_names[anno_i]
        gt_labels.append(objects.index(obj_name))
        gt_Rs.append(Rs[anno_i])
        gt_ts.append(transes[anno_i])

    est_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]
    gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]

    vis_im = img.copy()
    
    mask_est = np.zeros(img.shape[:2], dtype=bool)
    mask_gt = np.zeros(img.shape[:2], dtype=bool)
    
    if est_labels:
        for est_label, est_pose in zip(est_labels, est_poses):
            ren.render([est_label], [est_pose], K=K, seg_tensor=seg_tensor)
            est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
            est_edge = get_edge(est_mask, bw=3, out_channel=1)
            mask_est[est_edge != 0] = True

    if gt_labels:
        for gt_label, gt_pose in zip(gt_labels, gt_poses):
            ren.render([gt_label], [gt_pose], K=K, seg_tensor=seg_tensor)
            gt_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
            gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
            mask_gt[gt_edge != 0] = True

    green_color = np.array(mmcv.color_val("green"))
    red_color = np.array(mmcv.color_val("red"))

    vis_im[mask_gt] = red_color
    vis_im[mask_est] = green_color

    if est_Rs:
        axis_length = 0.05
        points_3d = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
        K_float = K.astype(np.float32)
        dist_coeffs = np.zeros((4, 1))

        for R_est, t_est in zip(est_Rs, est_ts):
            points_2d, _ = cv2.projectPoints(points_3d, R_est, t_est, K_float, dist_coeffs)
            points_2d = points_2d.reshape(-1, 2).astype(int)

            origin_2d = tuple(points_2d[0])
            x_axis_2d = tuple(points_2d[1])
            y_axis_2d = tuple(points_2d[2])
            z_axis_2d = tuple(points_2d[3])

            cv2.line(vis_im, origin_2d, x_axis_2d, (0, 0, 255), 2)
            cv2.line(vis_im, origin_2d, y_axis_2d, (0, 255, 0), 2)
            cv2.line(vis_im, origin_2d, z_axis_2d, (255, 0, 0), 2)

    show = False
    if show:
        grid_show([img_bbox[:, :, ::-1], vis_im[:, :, ::-1]], ["im", "est"], row=1, col=2)
    else:
        current_obj_name = id2obj[item['obj_id']] 
        save_path_1 = osp.join(vis_dir, "{}_{}_{:06d}_vis_depth{}.png".format(current_obj_name, item['sensor'], im_id, item['depth']))
        mmcv.imwrite(vis_im, save_path_1)