# iterates all images in the SS6D dataset and runs inference using the PEM model

import gorilla
import argparse
import os
import sys
from PIL import Image
import os.path as osp
import numpy as np
import random
import importlib
import json
from tqdm import tqdm
import logging


import torch
import torchvision.transforms as transforms
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..', 'Pose_Estimation_Model')
sys.path.append(os.path.join(ROOT_DIR, 'provider'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")
    # pem
    # parser.add_argument("--gpus",
    #                     type=str,
    #                     default="1",
    #                     help="gpu ids to use, e.g. 0,1,2")
    parser.add_argument("--model",
                        type=str,
                        default="pose_estimation_model",
                        help="path to model file")
    parser.add_argument("--config",
                        type=str,
                        default="config/base.yaml",
                        help="path to config file, different config.yaml use different config")
    parser.add_argument("--iter",
                        type=int,
                        default=600000,
                        help="epoch num. for testing")
    parser.add_argument("--exp_id",
                        type=int,
                        default=0,
                        help="")
    
    # input
    parser.add_argument("--base_path", nargs="?", help="Path to test dir of SS6D")
    parser.add_argument("--template_path", help="rendered templates of obj")
    parser.add_argument("--obj_id", help="obj id of SS6D (e.g. 0 for spray)")
    parser.add_argument("--depth_level", help="depth level (e.g. 0, 1, 2, 3)")
    parser.add_argument("--det_score_thresh", default=0.2, help="The score threshold of detection")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args = get_parser()
    exp_name = args.model + '_' + \
        osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    # cfg.gpus     = args.gpus
    cfg.model_name = args.model
    cfg.log_dir  = log_dir
    cfg.test_iter = args.iter

    cfg.base_path = args.base_path
    cfg.template_path = args.template_path
    cfg.obj_id = int(args.obj_id)
    cfg.depth_level = int(args.depth_level)

    cfg.det_score_thresh = args.det_score_thresh
    # gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

    return  cfg



from data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from draw_utils import draw_detections
import pycocotools.mask as cocomask
import trimesh

rgb_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    rgb = Image.fromarray(np.uint8(rgb))
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat


def _get_template(path, cfg, tem_index=1):
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz


def get_templates(path, cfg):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, i)
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
    return all_tem, all_tem_pts, all_tem_choose


def get_test_data(rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh, cfg):
    dets = []


    # with open(seg_path) as f:
    #     dets_ = json.load(f) # keys: scene_id, image_id, category_id, bbox, score, segmentation
    # for det in dets_:
    #     if det['score'] > det_score_thresh:
    #         dets.append(det)
    # del dets_

    import heapq

    with open(seg_path) as f:
        dets_ = json.load(f)  # keys: scene_id, image_id, category_id, bbox, score, segmentation

    # 상위 30개 score 기준으로 선택
    dets = heapq.nlargest(30, dets_, key=lambda x: x['score'])


    # 숫자만 추출하고 마지막 숫자 하나만 사용
    import re
    image_name = os.path.basename(rgb_path)
    match = re.search(r'\d+', image_name)
    image_id = match.group(0)[-1]

    cam_info = json.load(open(cam_path))
    K = np.array(cam_info[image_id]['cam_K']).reshape(3, 3)

    whole_image = load_im(rgb_path).astype(np.uint8)
    if len(whole_image.shape)==2:
        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
    whole_depth = load_im(depth_path).astype(np.float32) * cam_info[image_id]['depth_scale'] / 1000.0
    whole_pts = get_point_cloud_from_depth(whole_depth, K)

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))


    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    for inst in dets:
        seg = inst['segmentation']
        score = inst['score']

        # mask
        h,w = seg['size']
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, whole_depth > 0)
        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # pts
        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            continue
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # rgb
        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(score)
        all_dets.append(inst)

    ret_dict = {}
    ret_dict['pts'] = torch.stack(all_cloud).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict['pts'].size(0)
    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets



if __name__ == "__main__":
    cfg = init()

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    # model
    print("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    model = model.cuda()
    model.eval()
    checkpoint = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'checkpoints', 'sam-6d-pem-base.pth')
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

    print("=> extracting templates ...")
    tem_path = cfg.template_path
    all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, cfg.test_dataset)
    with torch.no_grad():
        all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    cad_id = cfg.obj_id + 1
    cad_path = os.path.join(cfg.base_path, "models", f"obj_{cad_id:06d}.ply")

    for b_level in ["B5", "B25", "B50", "B75", "B100"]:
    # for b_level in ["B5"]:
        bright_path = os.path.join(cfg.base_path, "test", b_level, f"{cfg.obj_id:06d}")
        cam_path = os.path.join(bright_path, "scene_camera.json")
        for sensor in ["AE", "AEG16", "AEG48", "AEG80", "AEG112", "E9G16", "E9G48", "E9G80", "E9G112",
                        "E39G16", "E39G48", "E39G80", "E39G112", "E156G16", "E156G48", "E156G80", "E156G112",
                        "E625G16", "E625G48", "E625G80", "E625G112", "E2500G16", "E2500G48", "E2500G80", "E2500G112"]:
        # for sensor in ["E156G16", "E156G48", "E156G80", "E156G112", "E625G16", "E625G48", "E625G80", "E625G112", "E2500G16", "E2500G48", "E2500G80", "E2500G112"]:
        # for sensor in ["E2500G112"]:
            detections_all = []
            sensor_path = os.path.join(bright_path, "rgb", sensor)
            for filename in tqdm(sorted(os.listdir(sensor_path)), desc=f"Processing {b_level} {sensor}"):
                rgb_path = os.path.join(bright_path, "rgb", sensor, filename)
                depth_path = os.path.join(bright_path, "depth", str(cfg.depth_level), filename)
                output_dir = os.path.join(cfg.base_path, "outputs_whole_sam", f"obj_{cfg.obj_id}", b_level, f"depth_{cfg.depth_level}", sensor, filename.split('.')[0])
                os.makedirs(output_dir, exist_ok=True)
                seg_path = os.path.join(output_dir, "detection_ism.json")
                # seg_path = os.path.join(cfg.base_path, "outputs_whole_sam", f"obj_{cfg.obj_id}", b_level, f"depth_{cfg.depth_level}", sensor, filename.split('.')[0], "detection_ism.json")

                try:
                    if not os.path.exists(seg_path):
                        missing_path = f"obj_{cfg.obj_id}/{b_level}/depth_{cfg.depth_level}/{sensor}/{filename.split('.')[0]}/detection_ism.json"
                        raise FileNotFoundError(f"Instance Segmentation file not found: {missing_path}")
                    
                    # if not os.path.exists(os.path.join(output_dir, "failed_pem_OOM.json")):
                    #     logging.info(f"Skipping {filename} as it did not failed for OOM")
                    #     continue

                    if (os.path.exists(os.path.join(output_dir, "detection_pem.json"))) and (os.path.exists(os.path.join(output_dir, "vis_pem.png"))):
                        logging.info(f"Skipping {filename} as detection already exists")
                        continue
                    # elif (os.path.exists(os.path.join(output_dir, "failed_pem.json"))):
                    #     logging.info(f"Skipping {filename} as it failed previously")
                    #     continue
                        
                    # print("=> loading input data ...")

                    input_data, img, whole_pts, model_points, detections = get_test_data(
                        rgb_path, depth_path, cam_path, cad_path, seg_path, 
                        cfg.det_score_thresh, cfg.test_dataset
                    )
                    ninstance = input_data['pts'].size(0)
                    # print("DEBUGGING: ninstance = ", ninstance)
                    
                    # print("=> running model ...")
                    with torch.no_grad():
                        input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
                        input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)
                        out = model(input_data)

                    if 'pred_pose_score' in out.keys():
                        pose_scores = out['pred_pose_score'] * out['score']
                    else:
                        pose_scores = out['score']
                    pose_scores = pose_scores.detach().cpu().numpy()
                    pred_rot = out['pred_R'].detach().cpu().numpy()
                    pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

                    # print("=> saving results ...")
                    os.makedirs(f"{output_dir}", exist_ok=True)
                    for idx, det in enumerate(detections):
                        detections[idx]['score'] = float(pose_scores[idx])
                        detections[idx]['R'] = list(pred_rot[idx].tolist())
                        detections[idx]['t'] = list(pred_trans[idx].tolist())

                    with open(os.path.join(f"{output_dir}", 'detection_pem.json'), "w") as f:
                        json.dump(detections, f)

                    detections_all.extend(detections)

                    # print("=> visualizating ...")
                    save_path = os.path.join(f"{output_dir}", 'vis_pem.png')
                    valid_masks = pose_scores == pose_scores.max()
                    K = input_data['K'].detach().cpu().numpy()[valid_masks]
                    vis_img = visualize(img, pred_rot[valid_masks], pred_trans[valid_masks], model_points*1000, K, save_path)
                    vis_img.save(save_path)

                    del input_data
                    torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError as e:
                    print(f"RuntimeError processing {rgb_path}: {e}")
                    with open(os.path.join(output_dir, "failed_pem_OOM.json"), 'w') as f:
                        json.dump(str(e), f)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error processing {rgb_path}: {e}")
                    with open(os.path.join(output_dir, "failed_pem.json"), 'w') as f:
                        json.dump([], f)
                    torch.cuda.empty_cache()

            merged_save_path = os.path.join(cfg.base_path, "outputs_whole_sam", f"obj_{cfg.obj_id}", b_level, f"depth_{cfg.depth_level}", sensor)
            merged_path = os.path.join(merged_save_path, "merged_pem.json")
            os.makedirs(merged_save_path, exist_ok=True)

            if os.path.exists(merged_path):
                # 기존 결과가 있는 경우 불러오기
                with open(merged_path, "r") as f:
                    existing_data = json.load(f)

                # 기존 + 새로운 결과 합치기
                if detections_all:
                    merged_data = existing_data + detections_all
                else:
                    merged_data = existing_data
            else:
                merged_data = detections_all

            # 다시 저장
            with open(merged_path, "w") as f:
                json.dump(merged_data, f)