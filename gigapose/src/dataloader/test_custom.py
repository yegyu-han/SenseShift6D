from __future__ import annotations

# Standard Library
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Third Party
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# MegaPose
from src.utils.bbox import BoundingBox
import src.megapose.utils.tensor_collection as tc
from src.megapose.utils.tensor_collection import PandasTensorCollection
from src.megapose.datasets.scene_dataset_ss6d import SceneObservation, ObjectData
from src.custom_megapose.custom_scene_dataset import (
    CustomSceneDataset,
)

from src.custom_megapose.template_dataset import TemplateDataset, NearestTemplateFinder
from src.dataloader.keypoints import KeyPointSampler
from bop_toolkit_lib import inout
from src.dataloader.train import GigaPoseTrainSet
from src.utils.logging import get_logger
from src.utils.inout_custom import (
    load_test_list_and_cnos_detections,
    load_test_list_and_init_locs,
)
from bop_toolkit_lib import pycoco_utils
from src.utils.dataset import LMO_ID_to_index
from src.lib3d.numpy import matrix4x4

logger = get_logger(__name__)
ListBbox = List[int]
ListPose = List[List[float]]
SceneObservationTensorCollection = PandasTensorCollection

SingleDataJsonType = Union[str, float, ListPose, int, ListBbox, Any]
DataJsonType = Union[Dict[str, SingleDataJsonType], List[SingleDataJsonType]]

@dataclass
class GigaPoseTestSet(GigaPoseTrainSet):
    def __init__(
        self,
        batch_size,
        root_dir,
        dataset_name,
        depth_scale,
        template_config,
        transforms,
        test_setting,
        load_gt=True,
        init_loc_path=None,  # for refinement
        brightness="B50",
        rgb_sensor="AE",
        depth_sensor="0"
    ):
        split, model_name = self.get_split_name(dataset_name)
        self.batch_size = batch_size
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.brightness = brightness
        self.rgb_sensor = rgb_sensor
        self.depth_sensor = depth_sensor
        if self.transforms.rgb_augmentation:
            self.transforms.rgb_transform.transform = [
                transform for transform in self.transforms.rgb_transform.transform
            ]

        # load the dataset
        dataset_dir = self.root_dir / self.dataset_name
        self.scene_dataset = CustomSceneDataset(dataset_dir / split, depth_scale=depth_scale, brightness=self.brightness, rgb_sensor=self.rgb_sensor, depth_sensor=self.depth_sensor)


        # load the template dataset
        model_infos = inout.load_json(
            self.root_dir / self.dataset_name / model_name / "models_info.json"
        )
        model_infos = [{"obj_id": int(obj_id)} for obj_id in model_infos.keys()]

        template_config.dir += f"/{dataset_name}"
        self.template_dataset = TemplateDataset.from_config(
            model_infos, template_config
        )
        self.template_finder = NearestTemplateFinder(template_config)

        # depending on setting:
        # 1. localization: load target_objects
        # 2. detection: not target_objects
        assert test_setting in [
            "localization",
            "detection",
        ], f"{test_setting} not supported!"
        self.load_detections(test_setting=test_setting)
        
        # --- 📌 [추가] 데이터 소스 동기화 및 필터링 ---
        # 1. cnos_dets(탐지 정보)에 있는 모든 image_key를 집합(set)으로 만듭니다.
        #    이렇게 하면 조회가 매우 빨라집니다.
        valid_image_keys = set(self.cnos_dets.keys())

        # 2. scene_dataset의 데이터 목록(frame_index)을 가져옵니다.
        scene_df = self.scene_dataset.frame_index

        # 3. scene_dataset의 각 행에 대해 cnos_dets와 동일한 형식의 image_key를 생성합니다.
        scene_df['image_key'] = scene_df.apply(
            lambda row: f"{int(row['scene_id']):06d}_{int(row['view_id']):06d}", axis=1
        )

        # 4. 탐지 정보가 존재하는 image_key를 가진 행만 남깁니다.
        filtered_df = scene_df[scene_df['image_key'].isin(valid_image_keys)].copy()
        
        # 5. 필터링된 데이터프레임으로 scene_dataset의 목록을 교체합니다.
        self.scene_dataset.frame_index = filtered_df
        logger.info(f"Filtered scene_dataset: {len(scene_df)} -> {len(filtered_df)} images with detections.")
        # -----------------------------------------------

        if init_loc_path is not None:
            self.load_init_loc(init_loc_path, test_setting)
            logger.info("Loaded init loc for refinement!")
        self.load_gt = load_gt

        # keypoint sampler
        self.keypoint_sampler = KeyPointSampler()

    def load_detections(self, test_setting):
        if test_setting == "localization":
            max_det_per_object_id = 32 if self.dataset_name == "icbin" else 16
        else:
            max_det_per_object_id = None
        self.test_list, self.cnos_dets = load_test_list_and_cnos_detections(
            self.root_dir,
            self.dataset_name,
            test_setting,
            max_det_per_object_id=max_det_per_object_id,
            brightness = self.brightness,
            rgb_sensor = self.rgb_sensor,
            depth_sensor = self.depth_sensor
        )

    def load_init_loc(self, init_loc_path, test_setting, min_score=0.25):
        (
            self.test_list,
            self.init_locs,
            self.num_hypothesis,
        ) = load_test_list_and_init_locs(
            self.root_dir, self.dataset_name, init_loc_path, test_setting
        )
        new_init_locs = {}
        init_number_locs, filtered_number_locs = 0, 0
        # drop instance_id having low confidence score
        for image_key in tqdm(self.init_locs, desc="Filtering init locs"):
            image_locs = self.init_locs[image_key]
            init_number_locs += len(image_locs)
            # group by instance_id
            image_locs_by_instance_id = {}
            for idx, loc in enumerate(image_locs):
                # 'instance_id'가 있으면 사용하고, 없으면 이미지 내 순서(idx)를 ID로 사용
                instance_id = loc.get("instance_id", idx)
                if instance_id not in image_locs_by_instance_id:
                    image_locs_by_instance_id[instance_id] = []
                image_locs_by_instance_id[instance_id].append(loc)

            new_image_locs = []
            for instance_id in image_locs_by_instance_id:
                locs = image_locs_by_instance_id[instance_id]
                best_score = max([loc["score"] for loc in locs])
                if best_score < min_score:
                    continue
                new_image_locs.extend(locs)
                filtered_number_locs += len(locs)
            if len(new_image_locs) == 0:
                logger.warning(f"Image key {image_key} has no locs!")
                new_image_locs = image_locs
            new_init_locs[image_key] = new_image_locs
        self.init_locs = new_init_locs
        logger.info(f"Drop from {init_number_locs} to {filtered_number_locs} locs!")
        if self.num_hypothesis == 1:
            self.instance_id_counter = 0
        logger.info(f"Loaded {self.num_hypothesis} init locs!")

    def get_split_name(self, dataset_name):
        if dataset_name in ["hb", "tless"]:
            split = "test_primesense"
        else:
            split = "test"
        logger.info(f"Split: {split} for {dataset_name}!")
        if dataset_name in ["tless"]:
            model_name = "models_cad"
        else:
            model_name = "models"
        return split, model_name

    def load_test_list(self, batch: List[SceneObservation]):
        target_lists = []
        detection_times = []
        for scene_obs in batch:
            infos = scene_obs.infos
            scene_id, im_id = int(infos.scene_id), int(infos.view_id)
            image_key = f"{scene_id:06d}_{im_id:06d}"
            target_list = self.test_list[image_key]
            target_lists.append(target_list)
            if image_key in self.cnos_dets:
                detection_times.append(self.cnos_dets[image_key][0]["time"])
            else:
                logger.warning(f"Image key {image_key} not in cnos_dets!")
                detection_times.append(0)
        assert len(target_lists) == len(detection_times)

        target_list_dict = {
            "im_id": [],
            "scene_id": [],
            "obj_id": [],
            "inst_count": [],
            "detection_time": [],
        }
        for idx, target_list in enumerate(target_lists):
            if len(target_list) == 0:
                continue
            for target in target_list:
                for name in target_list_dict:
                    if name == "detection_time":
                        continue
                    if name == "obj_id" and "lmo" in self.dataset_name:
                        target_list_dict[name].append(LMO_ID_to_index[target[name]])
                    else:
                        target_list_dict[name].append(target[name])
                target_list_dict["detection_time"].append(detection_times[idx])
        target_list_pd = pd.DataFrame(target_list_dict)
        return tc.PandasTensorCollection(infos=target_list_pd)

    def add_detections(self, batch: List[SceneObservation]):
        for scene_obs in batch:
            infos = scene_obs.infos
            scene_id, im_id = int(infos.scene_id), int(infos.view_id)
            image_key = f"{scene_id:06d}_{im_id:06d}"
            dets = self.cnos_dets[image_key]
            if len(scene_obs.object_datas) == 0:
                gt_available = False
            else:
                gt_available = True
                gt_mapping = {
                    obj_data.label: obj_data for obj_data in scene_obs.object_datas
                }
            object_datas = []
            binary_masks = {}
            for idx, det in enumerate(dets):
                data = {}
                # use confidence score as visibility
                data["visib_fract"] = det["score"]
                data["bbox_modal"] = det["bbox"]
                data["bbox_amodal"] = det["bbox"]
                data["label"] = str(det["category_id"])
                data["unique_id"] = f"{idx+1}"

                if gt_available and data["label"] in gt_mapping:
                    obj_data = gt_mapping[data["label"]]
                    data["TWO"] = obj_data.TWO._T
                else:
                    data["TWO"] = [[1, 0, 0, 0], [0, 0, 0]]
                object_data = ObjectData.from_json(data)
                object_datas.append(object_data)

                # load mask
                binary_mask = pycoco_utils.rle_to_binary_mask(det["segmentation"])
                binary_masks[object_data.unique_id] = binary_mask
            scene_obs.object_datas = object_datas
            scene_obs.binary_masks = binary_masks
        return batch

    def double_check_test_list(self, real_data, test_list):
        labels = np.asarray(real_data.infos.label).astype(np.int32)
        test_obj_ids = test_list.infos.obj_id
        assert np.allclose(np.unique(labels), np.unique(test_obj_ids))

    def process_real(self, batch, test_mode=False, min_box_size=50):
        # get the ground truth data
        rgb = batch["rgb"] / 255.0
        depth = batch["depth"].squeeze(1)
        detections = batch["gt_detections"]
        data = batch["gt_data"]

        # make bounding box square
        bboxes = BoundingBox(detections.bboxes, "xywh")
        if test_mode:
            idx_selected = np.arange(len(detections.bboxes))
        else:
            # keep only valid bounding boxes
            idx_selected = np.random.choice(
                np.arange(len(detections.bboxes)),
                min(self.batch_size, len(detections.bboxes)),
                replace=False,
            )

        bboxes = bboxes.reset(idx_selected)
        batch_im_id = detections[idx_selected].infos.batch_im_id
        masks = data.masks[idx_selected]
        K = data.K[idx_selected].float()
        pose = data.TWO[idx_selected].float()

        rgb = rgb[batch_im_id]
        depth = depth[batch_im_id]
        m_rgb = rgb * masks[:, None, :, :]

        m_rgba = torch.cat([m_rgb, masks[:, None, :, :]], dim=1)
        cropped_data = self.transforms.crop_transform(bboxes.xyxy_box, images=m_rgba)

        out_data = tc.PandasTensorCollection(
            full_rgb=rgb,
            full_depth=depth,
            K=K,
            rgb=cropped_data["images"][:, :3],
            mask=cropped_data["images"][:, -1],
            M=cropped_data["M"],
            pose=pose,
            infos=data[idx_selected].infos,
        )

        return out_data

    def collate_fn(
        self,
        batch: List[SceneObservation],
    ):
        gt_available = False if self.dataset_name in ["hb", "itodd"] else True
        load_gt = gt_available and self.load_gt

        # load test list
        test_list = self.load_test_list(batch)

        # add detections for each scene id
        if not load_gt:
            batch = self.add_detections(batch)

        # convert to tensor collection
        batch = SceneObservation.collate_fn(batch)

        # load real data
        real_data = self.process_real(batch, test_mode=True)

        if load_gt:
            # load template data
            template_data, T_real2temp, T_temp2real = self.process_template(real_data)

            # compute keypoints, template is source, real is target
            keypoints, rel_data = self.process_keypoints(
                real_data, template_data, T_real2temp, T_temp2real
            )

            # --- [디버깅 코드] keypoints 확인 ---
            # print("[DEBUG] keypoints dictionary content:")
            # for key, value in keypoints.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"  - {key}: shape={value.shape}, min={value.min():.2f}, max={value.max():.2f}, mean={value.mean():.2f}")
            
            if "lmo" in self.dataset_name:
                # workaround for indexing LMO: update object id to be in range(8)
                new_labels = real_data.infos.label
                new_labels = [str(LMO_ID_to_index[int(label)]) for label in new_labels]
                real_data.infos.label = new_labels

            out_data = tc.PandasTensorCollection(
                src_img=self.transforms.normalize(template_data.rgb),
                src_mask=template_data.mask,
                src_K=template_data.K,
                src_M=template_data.M,
                src_pts=keypoints["src_pts"],
                tar_img=self.transforms.normalize(real_data.rgb),
                tar_mask=real_data.mask,
                tar_K=real_data.K,
                tar_M=real_data.M,
                tar_pose=real_data.pose,
                tar_pts=keypoints["tar_pts"],
                relScale=rel_data["relScale"],
                relInplane=rel_data["relInplane"],
                infos=real_data.infos,
                test_list=test_list,
            )
        else:
            if "lmo" in self.dataset_name:
                # workaround for indexing LMO: update object id to be in range(8)
                new_labels = real_data.infos.label
                new_labels = [str(LMO_ID_to_index[int(label)]) for label in new_labels]
                real_data.infos.label = new_labels

            out_data = tc.PandasTensorCollection(
                tar_img=self.transforms.normalize(real_data.rgb),
                tar_mask=real_data.mask,
                tar_K=real_data.K,
                tar_M=real_data.M,
                infos=real_data.infos,
                test_list=test_list,
            )
        if not load_gt:
            self.double_check_test_list(real_data, test_list)
        return out_data

    def collate_refine_fn(self, batch: List[SceneObservation]):
        assert len(batch) == 1, "Only support batch size 1 for refinement!"
        scene_obs = batch[0]
        rgb = torch.from_numpy(scene_obs.rgb / 255.0)
        rgb = rgb.permute(2, 0, 1)
        rgb = rgb.unsqueeze(0)
        K = torch.from_numpy(scene_obs.camera_data.K)
        K = K.unsqueeze(0)

        infos = scene_obs.infos
        scene_id, im_id = int(infos.scene_id), int(infos.view_id)
        image_key = f"{scene_id:06d}_{im_id:06d}"
        if image_key not in self.init_locs:
            logger.warning(f"Image key {image_key} not in init locs!")
            out_data = tc.PandasTensorCollection(
                rgb=rgb.float(),
                K=K.float(),
                infos=pd.DataFrame(),
            )
        else:
            init_locs = self.init_locs[image_key]
            names = [
                "scene_id",
                "im_id",
                "obj_id",
                "instance_id",
                "score",
                "TCO_init",
                "time",
            ]
            data = {name: [] for name in names}
            for idx_loc, loc in enumerate(init_locs):
                for name in names:
                    if name == "TCO_init":
                        R, t = loc["R"], loc["t"]
                        TCO = matrix4x4(R, t, scale_translation=0.001)
                        data["TCO_init"].append(TCO)
                    # add instance_id in case of multiple hypothesis
                    elif name == "instance_id":
                        if self.num_hypothesis > 1:
                            instance_id = loc[name]
                        else:
                            instance_id = self.instance_id_counter
                            self.instance_id_counter += 1
                        data["instance_id"].append(instance_id)
                    else:
                        data[name].append(loc[name])
            for name in names:
                data[name] = np.asarray(data[name])

            data["TCO_init"] = torch.from_numpy(data["TCO_init"])

            # --- GT Pose와 추가 정보를 여기에 추가 ---
            gt_datas = scene_obs.object_datas
            gt_poses_map = {int(d.label): d.TWO.matrix for d in gt_datas}
            
            gt_poses_list = []
            for obj_id in data["obj_id"]:
                gt_poses_list.append(gt_poses_map.get(obj_id, np.eye(4))) # GT가 없으면 단위행렬

            # infos DataFrame에 brightness, sensor 정보 추가
            infos_df = pd.DataFrame(
                dict(
                    scene_id=data["scene_id"],
                    im_id=data["im_id"],
                    matching_score=data["score"],
                    batch_im_id=np.zeros_like(data["instance_id"]).astype(np.int32),
                    instance_id=data["instance_id"].astype(np.int32),
                    label=[f"obj_{obj_id:06d}" for obj_id in data["obj_id"]],
                    # scene_obs에서 직접 정보를 가져와서 init_locs 개수만큼 반복
                    brightness=[scene_obs.infos.brightness] * len(data["obj_id"]),
                    rgb_sensor=[scene_obs.infos.rgb_sensor] * len(data["obj_id"]),
                    depth_sensor=[scene_obs.infos.depth_sensor] * len(data["obj_id"]),
                    time=data["time"],
                )
            )

            out_data = tc.PandasTensorCollection(
                rgb=rgb.float(),
                K=K.float(),
                TCO_init=data["TCO_init"].float(),
                gt_poses=torch.from_numpy(np.stack(gt_poses_list)).float(), # GT Pose 추가
                infos=infos_df, # 확장된 infos 사용
            )
        return out_data

    def collate_refine_fn_v2(self, batch: List[SceneObservation]):
        assert len(batch) == 1, "Only support batch size 1 for refinement!"
        scene_obs = batch[0]
        rgb = torch.from_numpy(scene_obs.rgb / 255.0)
        rgb = rgb.permute(2, 0, 1)
        rgb = rgb.unsqueeze(0)
        depth = torch.from_numpy(scene_obs.depth).unsqueeze(0).unsqueeze(0)
        K = torch.from_numpy(scene_obs.camera_data.K)
        K = K.unsqueeze(0)

        infos = scene_obs.infos
        scene_id, im_id = int(infos.scene_id), int(infos.view_id)
        image_key = f"{scene_id:06d}_{im_id:06d}"
        if image_key not in self.init_locs:
            logger.warning(f"Image key {image_key} not in init locs!")
            out_data = tc.PandasTensorCollection(
                rgb=rgb.float(),
                depth=depth.float(),
                K=K.float(),
                infos=pd.DataFrame(),
            )
        else:
            init_locs = self.init_locs[image_key]
            names = [
                "scene_id",
                "im_id",
                "obj_id",
                "instance_id",
                "score",
                "TCO_init",
                "time",
            ]
            data = {name: [] for name in names}
            for idx_loc, loc in enumerate(init_locs):
                for name in names:
                    if name == "TCO_init":
                        R, t = loc["R"], loc["t"]
                        TCO = matrix4x4(R, t, scale_translation=0.001)
                        data["TCO_init"].append(TCO)
                    # add instance_id in case of multiple hypothesis
                    elif name == "instance_id":
                        if self.num_hypothesis > 1:
                            instance_id = loc[name]
                        else:
                            instance_id = self.instance_id_counter
                            self.instance_id_counter += 1
                        data["instance_id"].append(instance_id)
                    else:
                        data[name].append(loc[name])
            for name in names:
                data[name] = np.asarray(data[name])

            data["TCO_init"] = torch.from_numpy(data["TCO_init"])

            # --- GT Pose와 추가 정보를 여기에 추가 ---
            gt_datas = scene_obs.object_datas
            gt_poses_map = {int(d.label): d.TWO.matrix for d in gt_datas}
            
            gt_poses_list = []
            for obj_id in data["obj_id"]:
                gt_poses_list.append(gt_poses_map.get(obj_id, np.eye(4))) # GT가 없으면 단위행렬

            # infos DataFrame에 brightness, sensor 정보 추가
            infos_df = pd.DataFrame(
                dict(
                    scene_id=data["scene_id"],
                    im_id=data["im_id"],
                    matching_score=data["score"],
                    batch_im_id=np.zeros_like(data["instance_id"]).astype(np.int32),
                    instance_id=data["instance_id"].astype(np.int32),
                    label=[f"obj_{obj_id:06d}" for obj_id in data["obj_id"]],
                    # scene_obs에서 직접 정보를 가져와서 init_locs 개수만큼 반복
                    brightness=[scene_obs.infos.brightness] * len(data["obj_id"]),
                    rgb_sensor=[scene_obs.infos.rgb_sensor] * len(data["obj_id"]),
                    depth_sensor=[scene_obs.infos.depth_sensor] * len(data["obj_id"]),
                    time=data["time"],
                )
            )

            out_data = tc.PandasTensorCollection(
                rgb=rgb.float(),
                depth=depth.float(),
                K=K.float(),
                TCO_init=data["TCO_init"].float(),
                gt_poses=torch.from_numpy(np.stack(gt_poses_list)).float(), # GT Pose 추가
                infos=infos_df, # 확장된 infos 사용
            )
        return out_data


if __name__ == "__main__":
    import time
    import torch
    from hydra.experimental import compose, initialize
    from torch.utils.data import DataLoader
    from src.megapose.datasets.scene_dataset import SceneObservation
    from src.libVis.torch import plot_keypoints_batch, plot_Kabsch
    from torchvision.utils import save_image
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from src.models.ransac import RANSAC
    import logging

    logging.basicConfig(level=logging.DEBUG)
    with initialize(config_path="../../configs/"):
        cfg = compose(config_name="train.yaml")
    OmegaConf.set_struct(cfg, False)

    init_loc_path = "/home/nguyen/Documents/datasets/gigaPose_datasets/results/large_tudlGigaPose/predictions/large-pbrreal-rgb-mmodel_tudl-test_tudlGigaPose.csv"
    save_dir = "./tmp"
    os.makedirs(save_dir, exist_ok=True)

    cfg.machine.batch_size = 1
    cfg.data.test.dataloader.batch_size = cfg.machine.batch_size
    cfg.data.test.dataloader.dataset_name = "ycbv"
    # cfg.data.test.dataloader.init_loc_path = init_loc_path
    test_dataset = instantiate(cfg.data.test.dataloader)

    dataloader = DataLoader(
        test_dataset.scene_dataset,
        batch_size=cfg.machine.batch_size,
        num_workers=10,
        collate_fn=test_dataset.collate_fn,
        # collate_fn=test_dataset.collate_refine_fn,
    )
    ransac = RANSAC(pixel_threshold=5)
    start_time = time.time()
    for idx, batch in enumerate(dataloader):
        # batch = batch.cuda()
        end_time = time.time()
        print(f"Time: {end_time - start_time}")
        start_time = end_time

        keypoint_img = plot_keypoints_batch(batch, concate_input_in_pred=False)
        batch.relScale = batch.relScale.unsqueeze(1).repeat(1, 256)
        batch.relInplane = batch.relInplane.unsqueeze(1).repeat(1, 256)

        # vis relative scale and inplane
        M, idx_failed, _ = ransac(batch)
        wrap_img = plot_Kabsch(batch, M)
        vis_img = torch.cat([keypoint_img, wrap_img], dim=3)
        save_image(
            vis_img,
            os.path.join(save_dir, f"{idx:06d}.png"),
            nrow=4,
        )
