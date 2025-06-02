import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
import pandas as pd

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.vis_utils.colormap import colormap
from lib.utils.mask_utils import mask2bbox_xyxy, cocosegm2mask, get_edge
from core.utils.data_utils import read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.vis_utils.image import grid_show
from lib.vis_utils.image import vis_image_bboxes_cv2


import matplotlib.pyplot as plt
from PIL import Image

score_thr = 0.3
colors = colormap(rgb=False, maximum=255)

# object info
id2obj = {1: "spray", 2:"pringles", 3:"tincase"}
objects = list(id2obj.values())


def load_predicted_csv(fname):
    df = pd.read_csv(fname)
    info_list = df.to_dict("records")
    return info_list


def parse_Rt_in_csv(_item):
    return np.array([float(i) for i in _item.strip(" ").split(" ")])


# Camera info
width = 1280
height = 720

tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

model_dir = "datasets/BOP_DATASETS/SenseShift6D/models/"

model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]

ren = EGLRenderer(
    model_paths,
    vertex_scale=0.001,
    use_cache=True,
    width=width,
    height=height,
)

# --- setup (modify here) ---
obj_id = 2 # 0:spray, 1:pringles, 2:tincase

lv=4
d_list = [0, 3]

e=2500
g=80
eg=f"e{e}g{g}"
# eg=f"ae"

# ---

obj_name = id2obj[obj_id+1]
test_name = obj_name
if obj_id == 0:
    test_name = "spray-"

for depth in d_list:
    pred_path = osp.join(
        f"output/hipose/prediction/{obj_name}/lv{lv}/d{depth}/{eg}.csv"
    )
    vis_dir = f"output/hipose/visualization/{obj_name}/lv{lv}/{eg}_d{depth}"

    bbox_path = "datasets/BOP_DATASETS/SenseShift6D/test/test_bboxes/scene_gt_info_bboxes.json"

    mmcv.mkdir_or_exist(vis_dir)

    preds_csv = load_predicted_csv(pred_path)
    pred_bboxes = mmcv.load(bbox_path)
    preds = {}
    for item in preds_csv:
        im_key = "{}/{}".format(item["scene_id"], item["im_id"])
        item["time"] = float(item["time"])
        item["score"] = float(item["score"])
        item["R"] = parse_Rt_in_csv(item["R"]).reshape(3, 3)
        item["t"] = parse_Rt_in_csv(item["t"]) / 1000
        item["obj_name"] = id2obj[item["obj_id"]]
        if im_key not in preds:
            preds[im_key] = []
        preds[im_key].append(item)

    dataset_name = f"ss6d_{obj_name}_lv{lv}{eg}_d{depth}_te"
    print(dataset_name)
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
            print(scene_im_id, "not detected")
            continue
        cur_preds = preds[scene_im_id]
        cur_bboxes = pred_bboxes[scene_im_id]
        kpts_2d_est = []
        est_Rs = []
        est_ts = []
        est_labels = []
        for pred_i, pred in enumerate(cur_preds):
            try:
                R_est = pred["R"]
                t_est = pred["t"]
                score = pred["score"]
                obj_name = pred["obj_name"]
            except:
                continue
            if score < score_thr:
                continue

            est_Rs.append(R_est)
            est_ts.append(t_est)
            est_labels.append(objects.index(obj_name))  # 0-based label

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

        img_bbox = vis_image_bboxes_cv2(
            img,
            bboxes,
            labels,
        )

        im_gray = mmcv.bgr2gray(img, keepdim=True)
        im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

        gt_Rs = []
        gt_ts = []
        gt_labels = []

        # 0-based label
        annos = d["annotations"]
        cat_ids = [anno["category_id"] for anno in annos]
        obj_names = [objs[cat_id] for cat_id in cat_ids]

        quats = [anno["quat"] for anno in annos]
        transes = [anno["trans"] for anno in annos]
        Rs = [quat2mat(quat) for quat in quats]
        for anno_i, anno in enumerate(annos):
            obj_name = obj_names[anno_i]
            gt_labels.append(objects.index(obj_name))  # 0-based label

            gt_Rs.append(Rs[anno_i])
            gt_ts.append(transes[anno_i])

        est_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]
        gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]

        ren.render(
            est_labels,
            est_poses,
            K=K,
            image_tensor=image_tensor,
            background=im_gray_3,
        )
        ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

        for gt_label, gt_pose in zip(gt_labels, gt_poses):
            ren.render([gt_label], [gt_pose], K=K, seg_tensor=seg_tensor)
            gt_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
            gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
            ren_bgr[gt_edge != 0] = np.array(mmcv.color_val("red"))

        for est_label, est_pose in zip(est_labels, est_poses):
            ren.render([est_label], [est_pose], K=K, seg_tensor=seg_tensor)
            # ren.render([est_label], [est_pose], K=K, image_tensor = None, seg_tensor=seg_tensor)
            est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
            est_edge = get_edge(est_mask, bw=3, out_channel=1)
            ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

        # --- no CAD model ---  
        mask = (est_edge == 0) & (gt_edge == 0)
        ren_bgr[mask] = img[mask]
        # --- no CAD model ---  

        vis_im = ren_bgr
        # save_path_1 = osp.join(vis_dir, "CAD/{}_{:06d}_vis1_CAD.png".format(scene_id, im_id))
        save_path_1 = osp.join(vis_dir, "{}_{:06d}_vis1.png".format(scene_id, im_id))
        mmcv.imwrite(vis_im, save_path_1)
        


        # === Add this block at the end of your loop to generate side-by-side comparisons ===
        # d0과 d1(d2,d3) 이미지를 비교해서 e{e}g{g}_compare 디렉토리에 저장
        if depth == d_list[0]:  
            continue
        print("Comparing images...")

        base_dir = f"output/hipose/visualization/{obj_name}/lv{lv}"
        # 비교 이미지 저장 경로
        compare_dir = osp.join(base_dir, f"compare_d{d_list[0]}&d{d_list[1]}")
        mmcv.mkdir_or_exist(compare_dir)

        # 이미지 경로 설정
        img_filename = f"{scene_id}_{im_id:06d}_vis1.png"
        img_path_d0 = osp.join(base_dir, f"{eg}_d0", img_filename)
        img_path_d1 = osp.join(base_dir, f"{eg}_d{d_list[1]}", img_filename)

        # 두 파일이 모두 존재할 때만 비교 이미지 생성
        if osp.exists(img_path_d0) and osp.exists(img_path_d1):
            im_d0 = Image.open(img_path_d0)
            im_d1 = Image.open(img_path_d1)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(im_d0)
            axes[0].set_title(f"id{im_id}_{eg}_d0")
            axes[0].axis("off")

            axes[1].imshow(im_d1)
            axes[1].set_title(f"id{im_id}_{eg}_d{d_list[1]}")
            axes[1].axis("off")

            plt.tight_layout()
            save_path_cmp = osp.join(compare_dir, f"{im_id:02d}_e{e}g{g}_cpr.png")
            fig.savefig(save_path_cmp)
            plt.close(fig)
