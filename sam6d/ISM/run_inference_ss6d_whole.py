# iterates all images in the SS6D dataset and runs inference using the ISM model
# use demo_ss6d_whole.sh to run this script

from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio.v2 as imageio
import distinctipy
import json
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    best_det = detections[0]
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def batch_input_data(depth_path, cam_path, device, image_id):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info[image_id]['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info[image_id]['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def initialize_model(segmentor_model="fastsam", template_path=None, stability_score_thresh=None):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
    return model, device

def run_inference(segmentor_model, base_path, template_path, obj_id, depth_level, stability_score_thresh):
    model, device = initialize_model(segmentor_model, template_path, stability_score_thresh)
    cad_id = obj_id + 1
    cad_path = os.path.join(base_path, "models", f"obj_{cad_id:06d}.ply")
    
    num_templates = len(glob.glob(f"{template_path}/*.npy"))
    boxes, masks, templates = [], [], []

    logging.info(f"Loading templates from {template_path}")
    logging.info(f"Number of templates: {num_templates}")
    
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_path, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_path, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))

    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )

    logging.info(f"Initializing reference objects (templates)")
    descriptors_path = os.path.join(template_path, "descriptors.pth")
    appe_descriptors_path = os.path.join(template_path, "descriptors_appe.pth")
    model.ref_data = {}

    # if descriptors and appe_descriptors already exist, load them
    if (not os.path.exists(descriptors_path) or not os.path.exists(appe_descriptors_path)):
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    if (os.path.exists(descriptors_path)):
        logging.info(f"Loading descriptors from {descriptors_path}")
        model.ref_data["descriptors"] = torch.load(descriptors_path).to(device)
    else:
        model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        torch.save(model.ref_data["descriptors"], descriptors_path)
    
    if (os.path.exists(appe_descriptors_path)): 
        logging.info(f"Loading appearance descriptors from {appe_descriptors_path}")
        model.ref_data["appe_descriptors"] = torch.load(appe_descriptors_path).to(device)
    else:
        model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]
                    ).unsqueeze(0).data
        torch.save(model.ref_data["appe_descriptors"], appe_descriptors_path)

    for b_level in ["B5", "B25", "B50", "B75", "B100"]:
        bright_path = os.path.join(base_path, "test", b_level, f"{obj_id:06d}")
        cam_path = os.path.join(bright_path, "scene_camera.json")
        # for sensor in ["AE","E9G16", "E9G48", "E9G80", "E9G112", "E39G16", "E39G48", "E39G80", "E39G112", "E156G16", "E156G48", "E156G80", "E156G112", "E625G16", "E625G48", "E625G80", "E625G112", "E2500G16", "E2500G48", "E2500G80", "E2500G112"]:
        # for sensor in ["E2500G112"]:
        # for sensor in ["AE","E9G16", "E9G48", "E9G80", "E9G112", "E39G16", "E39G48", "E39G80", "E39G112", "E156G16", "E156G48", "E156G80", "E156G112"]:
        for sensor in ["AE", "AEG16", "AEG48", "AEG80", "AEG112", "E9G16", "E9G48", "E9G80", "E9G112", "E39G16", "E39G48", "E39G80", "E39G112", "E156G16", "E156G48", "E156G80", "E156G112", "E625G16", "E625G48", "E625G80", "E625G112", "E2500G16", "E2500G48", "E2500G80", "E2500G112"]:
            detections_all = []
            sensor_path = os.path.join(bright_path, "rgb", sensor)
            for filename in tqdm(sorted(os.listdir(sensor_path)), desc=f"Processing {b_level} {sensor}"):
                rgb_path = os.path.join(bright_path, "rgb", sensor, filename)
                depth_path = os.path.join(bright_path, "depth", str(depth_level), filename)
                output_dir = os.path.join(base_path, "outputs_whole_sam", f"obj_{obj_id}", b_level, f"depth_{depth_level}", sensor, filename.split('.')[0])
                os.makedirs(output_dir, exist_ok=True)
                if (os.path.exists(os.path.join(output_dir, "detection_ism.npz"))) and (os.path.exists(os.path.join(output_dir, "detection_ism.json"))) and (os.path.exists(os.path.join(output_dir, "vis_ism.png"))):
                    logging.info(f"Skipping {filename} as detection already exists")
                    continue
                elif (os.path.exists(os.path.join(output_dir, "failed.json"))):
                    logging.info(f"Skipping {filename} as it failed previously")
                    continue

                try: 
                    # run inference (run propoals)
                    rgb = Image.open(rgb_path).convert("RGB")
                    detections = model.segmentor_model.generate_masks(np.array(rgb))
                    # init detections with masks and boxes
                    detections = Detections(detections)
                    # compute semantic descriptors and appearance descriptors for query proposals
                    with torch.no_grad():
                        query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

                        # matching descriptors
                        (
                            idx_selected_proposals,
                            pred_idx_objects,
                            semantic_score,
                            best_template,
                        ) = model.compute_semantic_score(query_decriptors)

                    # update detections with semantic score filtering
                    detections.filter(idx_selected_proposals)
                    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
                    if detections is None or len(detections) == 0:
                        raise ValueError("Detections are empty after filtering")

                    # compute the appearance score
                    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

                    image_name = os.path.basename(rgb_path)
                    image_id = os.path.splitext(image_name)[0]  # "000040"
                    image_id = str(int(image_id))  # "40"

                    batch = batch_input_data(depth_path, cam_path, device, image_id)
                    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
                    template_poses[:, :3, 3] *= 0.4
                    poses = torch.tensor(template_poses).to(torch.float32).to(device)
                    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

                    mesh = trimesh.load_mesh(cad_path)
                    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
                    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
                    
                    # compute the geometric score
                    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)
                    geometric_score, visible_ratio = model.compute_geometric_score(
                        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
                        )

                    # final score
                    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

                    detections.add_attribute("scores", final_score)
                    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
                    detections.to_numpy()

                    save_path = os.path.join(output_dir, "detection_ism")
                    detections.save_to_file(scene_id=0, frame_id=int(filename.split('.')[0]), runtime=0, file_path=save_path, dataset_name="SS6D", return_results=False)
                    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
                    save_json_bop23(save_path+".json", detections)
                    detections_all.append(detections)

                    vis_path = os.path.join(output_dir, "vis_ism.png")
                    vis_img = visualize(rgb, detections, vis_path)
                    vis_img.save(vis_path)

                    torch.cuda.empty_cache()
            
                except Exception as e:
                    logging.error(f"No valid detection for {rgb_path}: {e}")
                    with open(os.path.join(output_dir, "failed.json"), 'w') as f:
                        json.dump([], f)  # 빈 딕셔너리 형태로 초기화
                    continue

            # save all detections in a single json file for the current sensor setting
            merged_save_path = os.path.join(base_path, "outputs_whole_sam", f"obj_{obj_id}", b_level, f"depth_{depth_level}", sensor)
            os.makedirs(merged_save_path, exist_ok=True)
            merged_path = os.path.join(merged_save_path, "merged_detections.json")
            save_json_bop23(merged_path, detections_all)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--base_path", nargs="?", help="Path to test dir of SS6D")
    parser.add_argument("--template_path", help="rendered templates of obj")
    parser.add_argument("--obj_id", help="obj id of SS6D (e.g. 0 for spray)")
    parser.add_argument("--depth_level", help="depth level (e.g. 0, 1, 2, 3)")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of fastsam")
    args = parser.parse_args()
    run_inference(
        args.segmentor_model, args.base_path, args.template_path, int(args.obj_id), args.depth_level,
        stability_score_thresh=args.stability_score_thresh,
    )