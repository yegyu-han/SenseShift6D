# adapted from https://github.com/megapose6d/megapose6d/blob/master/src/megapose/datasets/web_scene_dataset.py#L1
import os
import cv2
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from PIL import Image
from bop_toolkit_lib import inout
from src.megapose.lib3d.transform import Transform
from src.utils.logging import get_logger
from src.megapose.datasets.scene_dataset_ss6d import (
    CameraData,
    ObjectData,
    ObservationInfos,
    SceneDataset,
    SceneObservation,
)

logger = get_logger(__name__)

def load_scene_ds_obs(
    sample: pd.Series,
    depth_scale: float = 1000,
    load_depth: bool = False,
    label_format: str = "{label}",
) -> SceneObservation:
    # load
    rgb = cv2.imread(sample['rgb_path'])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Image.open은 경로가 Path 객체일 때 더 안정적일 수 있습니다.
    depth = np.asarray(Image.open(Path(sample['depth_path'])))
    depth = depth.astype(np.float32)
    depth /= depth_scale

    cam_R_w2c = np.eye(3)
    cam_t_w2c = np.zeros(3)
    camera_data = CameraData(
        K=np.array(sample['camera']["cam_K"]).reshape(3, 3),
        TWC=Transform(
            cam_R_w2c,
            cam_t_w2c,
        ),
        resolution=rgb.shape[:2],
    )

    infos = ObservationInfos(scene_id=sample['scene_id'], view_id=sample['view_id'], brightness=sample['brightness'], rgb_sensor=sample['rgb_sensor'], depth_sensor=sample['depth_sensor'])

    object_datas = []
    # binary_masks 딕셔너리를 미리 초기화
    binary_masks_dict = {}

    # GT 데이터가 있는지 확인
    if 'gt' in sample and sample['gt'] is not None:
        for idx_obj, data in enumerate(sample['gt']):
            data_info = sample['gt_info'][idx_obj]
            data["visib_fract"] = data_info["visib_fract"]
            if data["visib_fract"] <= 0.1:
                continue
            data["bbox_modal"] = data_info["bbox_visib"]
            data["bbox_amodal"] = data_info["bbox_obj"]

            obj_id = data["obj_id"]
            unique_id = idx_obj + 1
            data["label"] = f"{obj_id}"
            data["unique_id"] = str(unique_id)
            rot = np.array(data["cam_R_m2c"]).reshape(3, 3)
            quat = R.from_matrix(rot).as_quat()
            trans = data["cam_t_m2c"]
            data["TWO"] = [quat.tolist(), np.array(trans).tolist()]

            object_data = ObjectData.from_json(data)
            object_datas.append(object_data)

            # --- 마스크 처리 로직을 루프 안으로 이동 ---
            # 현재 객체(idx_obj)에 해당하는 마스크 경로를 가져옵니다.
            mask_path = sample['mask_visib_paths'][idx_obj]
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            binary_mask = (mask_image > 0)
            
            # object_data의 idx_obj와 동일한 키를 사용합니다.
            binary_masks_dict[str(unique_id)] = binary_mask
            
    return SceneObservation(
        rgb=rgb,
        depth=depth,
        infos=infos,
        object_datas=object_datas,
        camera_data=camera_data,
        binary_masks=binary_masks_dict,
    )


class CustomSceneDataset(SceneDataset):
    def __init__(
        self,
        target_dir: Path,
        depth_scale: float = 1000.0,
        load_depth: bool = True,
        load_segmentation: bool = True,
        label_format: str = "{label}",
        load_frame_index: bool = False,
        brightness = "B50",
        rgb_sensor = "AE",
        depth_sensor = "0"
    ):
        self.depth_scale = depth_scale
        self.label_format = label_format
        self.target_dir = target_dir
        self.brightness = brightness
        self.rgb_sensor = rgb_sensor
        self.depth_sensor = depth_sensor

        frame_index = self.load_frame_index()

        super().__init__(
            frame_index=frame_index,
            load_depth=load_depth,
            load_segmentation=load_segmentation,
        )

    def load_frame_index(self) -> pd.DataFrame:
        target_dir = self.target_dir
        
        # 모든 샘플 정보를 담을 단일 리스트
        all_samples = []

        if(os.path.exists(target_dir)):
            for brightness_dir in os.listdir(target_dir): # B5
                if brightness_dir != self.brightness:
                    continue
                brightness_path = os.path.join(target_dir, brightness_dir)
                if not os.path.isdir(brightness_path): continue

                for scene_dir_name in os.listdir(brightness_path): # 000000
                    current_dir = os.path.join(brightness_path, scene_dir_name)
                    if not os.path.isdir(current_dir): continue

                    scene_camera_path = os.path.join(current_dir, "scene_camera.json")
                    if os.path.exists(scene_camera_path):
                        scene_params = inout.load_scene_camera(scene_camera_path)            
                        scene_gt_fn = os.path.join(current_dir, "scene_gt.json")
                        scene_gt_info_fn = os.path.join(current_dir, "scene_gt_info.json")
                        
                        scene_gts, scene_gt_infos = None, None
                        if os.path.exists(scene_gt_fn) and os.path.exists(scene_gt_info_fn):
                            scene_gts = inout.load_scene_gt(scene_gt_fn)
                            scene_gt_infos = inout.load_scene_gt(scene_gt_info_fn)
            
                        for img_key_str in sorted(scene_params.keys()):
                            img_id = int(img_key_str)
                            
                            rgb_base_path = os.path.join(current_dir, "rgb")
                            depth_base_path = os.path.join(current_dir, "depth")
                            
                            # 모든 RGB/Depth 센서 조합에 대해 루프
                            for rgb_sensor_dir in os.listdir(rgb_base_path):
                                if rgb_sensor_dir != self.rgb_sensor:
                                    continue
                                for depth_sensor_dir in os.listdir(depth_base_path):
                                    if depth_sensor_dir != self.depth_sensor:
                                        continue
                                    rgb_fn = os.path.join(rgb_base_path, rgb_sensor_dir, f"{img_id:06d}.png")
                                    depth_fn = os.path.join(depth_base_path, depth_sensor_dir, f"{img_id:06d}.png")

                                    # 샘플 하나의 정보를 담는 딕셔너리 생성
                                    sample_info = {
                                        'rgb_path': rgb_fn,
                                        'depth_path': depth_fn,
                                        'camera': scene_params[img_key_str],
                                        'scene_id': scene_dir_name,
                                        'view_id': f"{img_id:06d}",
                                        'brightness': brightness_dir,
                                        'rgb_sensor': rgb_sensor_dir,
                                        'depth_sensor': depth_sensor_dir,
                                    }

                                    # GT 정보가 있을 경우 추가
                                    if scene_gts and scene_gt_infos:
                                        sample_info['gt'] = scene_gts[img_id]
                                        sample_info['gt_info'] = scene_gt_infos[img_id]

                                        # 마스크 정보 추가
                                        mask_visib_fns = []
                                        for counter, gt_obj in enumerate(scene_gts[img_id]):
                                            mask_visib_fn = os.path.join(current_dir, "mask_visib", f"{img_id:06d}_{counter:06d}.png")
                                            mask_visib_fns.append(mask_visib_fn)
                                        sample_info['mask_visib_paths'] = mask_visib_fns

                                    all_samples.append(sample_info)
        
        # 리스트가 비어있으면 빈 DataFrame 반환
        if not all_samples:
            return pd.DataFrame()

        # 딕셔너리 리스트를 하나의 DataFrame으로 변환하여 반환
        frame_index = pd.DataFrame(all_samples)
        return frame_index

    def __getitem__(self, idx: int) -> SceneObservation:
        assert self.frame_index is not None
        row = self.frame_index.iloc[idx]
        obs = load_scene_ds_obs(row, load_depth=self.load_depth)
        return obs
