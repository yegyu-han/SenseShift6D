import os, cv2
import errno
import shutil
import numpy as np
import os.path as osp
import json, sys
import sys
import pandas as pd
from tqdm import tqdm
from bop_toolkit_lib import inout, pose_error
from pycocotools import mask as mask_utils
from src.utils.dataset import LMO_index_to_ID, cnos_detections
from src.utils.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)
MAX_VALUES = 1e6


def Calculate_ADD_Error_BOP(R_GT,t_GT, R_predict, t_predict, vertices):
    t_GT = t_GT.reshape((3,1))
    t_predict = np.array(t_predict).reshape((3,1))

    return pose_error.add(R_predict, t_predict, R_GT, t_GT, vertices)

def get_root_project():
    return Path(__file__).absolute().parent.parent.parent


def append_lib(path):
    sys.path.append(os.path.join(path, "src"))


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def write_txt(path, list_files):
    with open(path, "w") as f:
        for idx in list_files:
            f.write(idx + "\n")
        f.close()


def open_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_json(path):
    with open(path, "r") as f:
        # info = yaml.load(f, Loader=yaml.CLoader)
        info = json.load(f)
    return info


def save_json(path, info):
    # save to json without sorting keys or changing format
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


def save_npz(path, info):
    np.savez_compressed(path, **info)


def casting_format_to_save_json(data):
    # casting for every keys in dict to list so that it can be saved as json
    for key in data.keys():
        if (
            isinstance(data[key][0], np.ndarray)
            or isinstance(data[key][0], np.float32)
            or isinstance(data[key][0], np.float64)
            or isinstance(data[key][0], np.int32)
            or isinstance(data[key][0], np.int64)
        ):
            data[key] = np.array(data[key]).tolist()
    return data


def convert_dict_to_dataframe(data_dict, column_names, convert_to_list=True):
    if convert_to_list:
        data_list = list(data_dict.items())
    else:
        data_list = data_dict
    df = pd.DataFrame(data_list, columns=column_names)
    return df


def combine(list_dict):
    output = {}
    for dict_ in list_dict:
        for field in dict_.keys():
            for name_data in dict_[field].keys():
                key = field + "_" + name_data
                assert key not in output.keys()
                output[key] = dict_[field][name_data]
    return output


def group_by_image_level(data, image_key="im_id"):
    # group the detections by scene_id and im_id
    data_per_image = {}
    for det in data:
        if isinstance(det, dict):
            dets = [det]
        else:
            dets = det
        for det in dets:
            scene_id, im_id = int(det["scene_id"]), int(det[image_key])
            key = f"{scene_id:06d}_{im_id:06d}"
            if key not in data_per_image:
                data_per_image[key] = []
            data_per_image[key].append(det)
    return data_per_image


def save_bop_results(path, results, additional_name=None):
    # https://github.com/thodan/bop_toolkit/blob/37d79c4c5fb027da92bc40f36b82ea9b7b197f1d/bop_toolkit_lib/inout.py#L292
    if additional_name is not None:
        lines = [f"scene_id,im_id,brightness,rgb_sensor,depth_sensor,obj_id,score,R,t,time,{additional_name}"]
    else:
        lines = ["scene_id,im_id,brightness,rgb_sensor,depth_sensor,obj_id,score,R,t,time"]
    for res in results:
        if "time" in res:
            run_time = res["time"]
        else:
            run_time = -1
        lines.append(
            "{scene_id},{im_id},{brightness},{rgb_sensor},{depth_sensor},{obj_id},{score},{R},{t},{time}".format(
                scene_id=res["scene_id"],
                im_id=res["im_id"],
                brightness=res["brightness"],
                rgb_sensor=res["rgb_sensor"],
                depth_sensor=res["depth_sensor"],
                obj_id=res["obj_id"],
                score=res["score"],
                R=" ".join(map(str, res["R"].flatten().tolist())),
                t=" ".join(map(str, res["t"].flatten().tolist())),
                # add=res["add"],
                # auc=res["auc"],
                time=run_time,
            )
        )
        if additional_name is not None:
            lines[-1] += ",{}".format(res[f"{additional_name}"])
    with open(path, "w") as f:
        f.write("\n".join(lines))


def load_bop_results(path, additional_name=None):
    # https://github.com/thodan/bop_toolkit/blob/37d79c4c5fb027da92bc40f36b82ea9b7b197f1d/bop_toolkit_lib/inout.py#L249
    results = []
    if additional_name is not None:
        header = f"scene_id,im_id,brightness,rgb_sensor,depth_sensor,obj_id,score,R,t,time,{additional_name}"
        length_line = 11
    else:
        header = "scene_id,im_id,brightness,rgb_sensor,depth_sensor,obj_id,score,R,t,time"
        length_line = 10
    with open(path, "r") as f:
        line_id = 0
        for line in f:
            line_id += 1
            if line_id == 1 and header in line:
                continue
            else:
                elems = line.split(",")
                if len(elems) != length_line:
                    raise ValueError(
                        "A line does not have {} comma-sep. elements: {}".format(
                            length_line, line
                        )
                    )

                result = {
                    "scene_id": int(elems[0]),
                    "im_id": int(elems[1]),
                    "brightness": str(elems[2]),
                    "rgb_sensor": str(elems[3]),
                    "depth_sensor": str(elems[4]),
                    "obj_id": int(elems[5]),
                    "score": float(elems[6]),
                    "R": np.array(
                        list(map(float, elems[7].split())), np.float64
                    ).reshape((3, 3)),
                    "t": np.array(
                        list(map(float, elems[8].split())), np.float64
                    ).reshape((3, 1)),
                    # "add": float(elems[9]),
                    # "auc": float(elems[10]),
                    "time": float(elems[9]),
                }
                if additional_name is not None:
                    result[additional_name] = float(elems[10])
                results.append(result)
    return results


def averaging_runtime_bop_results(path, has_instance_id=False):
    results = load_bop_results(path, has_instance_id)
    times = {}
    # calculate mean time for each scene_id and im_id
    for result in results:
        result_key = "{:06d}_{:06d}".format(result["scene_id"], result["im_id"])
        if result_key not in times.keys():
            times[result_key] = []
        times[result_key].append(result["time"])
    for key in times.keys():
        times[key] = np.mean(times[key])
    # replace time in results
    for result in results:
        result_key = "{:06d}_{:06d}".format(result["scene_id"], result["im_id"])
        result["time"] = times[result_key]
    # save to new file
    save_bop_results(path, results, has_instance_id)
    # logger.info(f"Averaged and saved predictions to {path}")


def calculate_runtime_per_image(results, is_refined):
    """
    Calculate the correct run_time for each image as in BOP challenge
    coarse_run_time: run_time = detection_time + total_time(all_batched_images)
    total_run_time: run_time = coarse_run_time + total_time(refinement)
    """
    # sort times by image_id
    if is_refined:
        time_names = ["time", "refinement_time"]
    else:
        time_names = ["detection_time", "time"]

    times = {}
    new_results = []
    counter = 0
    for result in results:
        result_key = "{:06d}_{:06d}".format(result["scene_id"], result["im_id"])
        if result_key not in times.keys():
            times[result_key] = {name: [] for name in time_names}
            times[result_key]["batch_id"] = []
        assert "batch_id" in result.keys(), f"batch_id is not in {result}"
        # make sure that detection_time and each batch is counted only once
        if result["batch_id"] not in times[result_key]["batch_id"]:
            times[result_key]["batch_id"].append(result["batch_id"])
            times[result_key]["time"].append(result["time"])
            if not is_refined:
                times[result_key]["detection_time"] = result["additional_time"]
            else:
                times[result_key]["refinement_time"].append(result["additional_time"])

        # delete the key additional_time and batch_id in result
        del result["additional_time"]
        del result["batch_id"]

    # calculate run_time for each image

    total_run_times = {}
    for key in times.keys():
        time = times[key]
        if not is_refined:
            total_run_time = time["detection_time"] + np.sum(time["time"])
        else:
            assert len(time["refinement_time"]) == len(time["batch_id"])
            total_run_time = np.sum(time["refinement_time"]) + np.sum(time["time"])
        total_run_times[key] = total_run_time

    # update the run_time for each image
    average_run_times = []
    for result in results:
        result_key = "{:06d}_{:06d}".format(result["scene_id"], result["im_id"])
        result["time"] = total_run_times[result_key]
        average_run_times.append(result["time"])
    logger.info(f"Average runtime per image: {np.mean(average_run_times):.3f} s")
    return results


def save_predictions_from_batched_predictions(
    prediction_dir,
    dataset_name,
    model_name,
    run_id,
    is_refined,
):
    list_files = [file for file in os.listdir(prediction_dir) if file.endswith(".npz")]
    list_files = sorted(list_files)

    name_additional_time = "detection_time" if not is_refined else "refinement_time"
    top1_predictions = []
    instance_id = 0

    model_dir = "/hdd/tgyoon/GiGaPose/gigapose/gigaPose_datasets/datasets/SENSESHIFT6D/models"

    for batch_id, file in tqdm(
        enumerate(list_files), desc="Formatting predictions ..."
    ):
        data = np.load(osp.join(prediction_dir, file))
        assert len(data["poses"].shape) in [3, 4]

        for idx_sample in range(len(data["im_id"])):
            obj_id = int(data["object_id"][idx_sample])
            mesh_path = f"{model_dir}/obj_{obj_id:06d}.ply"
            model_info = inout.load_json(f"{model_dir}/models_info.json")
            obj_diameter = model_info[str(obj_id)]['diameter']
            vertices = inout.load_ply(mesh_path)["pts"]
            is_multihypothesis = len(data["poses"].shape) == 4

            if is_multihypothesis:
                # k개 후보 중 첫 번째(가장 점수가 높은) 예측을 사용
                pose_predict = data["poses"][idx_sample, 0]
                score = data["scores"][idx_sample, 0]
            else:
                pose_predict = data["poses"][idx_sample]
                score = data["scores"][idx_sample]

            t_predict = pose_predict[:3, 3]
            R_predict = pose_predict[:3, :3]
            # t_GT = data["gt_poses"][idx_sample][:3, 3]
            # R_GT = data["gt_poses"][idx_sample][:3, :3]

            # adx_error = Calculate_ADD_Error_BOP(R_GT, t_GT, R_predict, t_predict, vertices)
            
            # th = np.linspace(0, 0.10, num=100)
            # sum_correct = 0
            # for t in th:
            #     if adx_error < obj_diameter*t:
            #         sum_correct = sum_correct + 1
            # auc_error = sum_correct/100
            # adx_error = 10000
            # auc_error = 0
            
            top1_prediction = dict(
                scene_id=int(data["scene_id"][idx_sample]),
                im_id=int(data["im_id"][idx_sample]),
                brightness=data["brightness"][idx_sample].decode('utf-8') if hasattr(data["brightness"][idx_sample], 'decode') else data["brightness"][idx_sample],
                rgb_sensor=data["rgb_sensor"][idx_sample].decode('utf-8') if hasattr(data["rgb_sensor"][idx_sample], 'decode') else data["rgb_sensor"][idx_sample],
                depth_sensor=data["depth_sensor"][idx_sample].decode('utf-8') if hasattr(data["depth_sensor"][idx_sample], 'decode') else data["depth_sensor"][idx_sample],
                obj_id=obj_id,
                score=score,
                t=t_predict.flatten(),
                R=R_predict.flatten(),
                # add=adx_error,
                # auc=auc_error,
                time=data["time"][idx_sample],
                additional_time=data[name_additional_time][idx_sample],
                batch_id=np.copy(batch_id),
            )
            assert (
                "batch_id" in top1_prediction.keys()
            ), f"batch_id is not in {top1_prediction}"
            top1_predictions.append(top1_prediction)
            top1_prediction["instance_id"] = instance_id
            instance_id += 1

    name_file = f"{model_name}-pbrreal-rgb-mmodel_{dataset_name}-test_{run_id}"
    save_path = osp.join(prediction_dir, f"{name_file}MultiHypothesis.csv")
    calculate_runtime_per_image(top1_predictions, is_refined=is_refined)
    save_bop_results(
        save_path,
        top1_predictions,
        additional_name=None,
    )
    logger.info(f"Saved predictions to {save_path}")


def generate_test_list(all_detections):
    all_target_list = {}
    for im_key in all_detections:
        # map detections to target_list
        im_id, scene_id = im_key.split("_")
        im_id, scene_id = int(im_id), int(scene_id)
        im_target = {}
        im_dets = all_detections[im_key]
        for det in im_dets:
            if "category_id" in det:
                obj_id = det["category_id"]
            elif "obj_id" in det:
                obj_id = det["obj_id"]
            else:
                raise ValueError("category_id or obj_id is not in the detection!")
            if obj_id not in im_target:
                im_target[obj_id] = 1
            else:
                im_target[obj_id] += 1
        im_target_list = []
        for obj_id in im_target.keys():
            im_target_list.append(
                {
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "obj_id": obj_id,
                    "inst_count": im_target[obj_id],
                }
            )
        all_target_list[im_key] = im_target_list
    return all_target_list


def load_test_list_and_cnos_detections_mix(
    root_dir, dataset_name, test_setting, max_det_per_object_id=None, 
    brightness="B50", rgb_sensor="AE", depth_sensor="0"
):
    """
    GT bbox/mask와 est score/time을 조합하여 cnos_dets를 생성하는 함수.
    """
    gt_bbox_path = root_dir / dataset_name / "test" / "test_bboxes" / "scene_gt_info_bboxes.json"
    bbox_base_path = Path("/ssd/sjkim/SAM-6D/Data/SS6D/outputs_whole_sam")
    
    if not bbox_base_path.exists() or not gt_bbox_path.exists():
        raise FileNotFoundError("Base detection or GT path not found.")
    
    with open(gt_bbox_path) as f:
        gt_bboxes_raw = json.load(f)

    all_detections_list = []
    
    # 1. 모든 객체 폴더를 순회하며 탐지(est) 결과 취합
    for obj_dir in os.listdir(bbox_base_path):
        json_path = bbox_base_path / obj_dir / brightness / f"depth_{depth_sensor}" / f"merged_ism_topscore_{rgb_sensor}.json"
        
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            detections_in_file = json.load(f)
            current_scene_id = int(obj_dir.split('_')[-1])
            for det in detections_in_file:
                det['scene_id'] = current_scene_id
            all_detections_list.extend(detections_in_file)

    # 2. 취합된 탐지 결과를 cnos_dets 포맷으로 변환
    all_dets_per_image = {}
    for det_raw in all_detections_list: # 📌 [수정] 단일 탐지 결과(det_raw)를 직접 처리
        scene_id = det_raw['scene_id']
        im_id = det_raw['image_id']
        
        # GT bbox 매칭 및 저장용 키 생성
        gt_key = f"{scene_id}/{im_id}"
        image_key = f"{scene_id:06d}_{im_id:06d}"
        
        # GT 정보가 없으면 해당 탐지는 건너뜀
        if gt_key not in gt_bboxes_raw:
            continue
        
        if image_key not in all_dets_per_image:
            all_dets_per_image[image_key] = []
        
        # est와 GT의 객체 순서가 같다고 가정하고, 첫 번째 객체 정보 사용
        gt_data = gt_bboxes_raw[gt_key][0]

        processed_det = {
            'scene_id': scene_id,
            'image_id': im_id,
            'score': det_raw['score'],            # score는 탐지(est) 결과 사용
            'bbox': tuple(gt_data['bbox_est']),  # bbox는 GT 사용
            'category_id': det_raw['category_id'], # category_id는 est 결과 사용
            'time': det_raw.get('time', 0.0)      # time은 탐지(est) 결과 사용
        }
        
        # GT 마스크 이미지 로드
        scene_folder_path = root_dir / dataset_name / "test" / brightness / f"{scene_id:06d}"
        mask_counter = 0 # 한 이미지에 객체가 하나라고 가정
        mask_path = scene_folder_path / "mask_visib" / f"{im_id:06d}_{mask_counter:06d}.png"

        if os.path.exists(mask_path):
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_image is not None:
                binary_mask = (mask_image > 0).astype(np.uint8)
                rle = binary_mask_to_rle(binary_mask)
                processed_det['segmentation'] = rle

        all_dets_per_image[image_key].append(processed_det)
        
    return (
        generate_test_list(all_dets_per_image),
        all_dets_per_image,
    )

def load_test_list_and_cnos_detections(
    root_dir, dataset_name, test_setting, max_det_per_object_id=None, 
    brightness="B50", rgb_sensor="AE", depth_sensor="0"
):
    """
    Ground Truth bbox와 mask를 불러와 cnos_dets 포맷으로 변환하는 함수.
    """
    # 1. GT BBox 파일 불러오기
    bbox_path = root_dir / dataset_name / "test" / "test_bboxes" / "scene_gt_info_bboxes.json"
    if not bbox_path.exists():
        raise FileNotFoundError(f"GT file not found at: {bbox_path}")

    with open(bbox_path) as f:
        all_detections_raw = json.load(f)

    # 2. 데이터를 cnos_dets 포맷으로 변환
    all_dets_per_image = {}
    # --- 📌 [수정] for 루프 시작 ---
    for key, dets_raw in all_detections_raw.items():
        scene_id, im_id = key.split('/')
        new_key = f"{int(scene_id):06d}_{int(im_id):06d}"
        
        processed_dets = []
        # 한 이미지에 여러 객체가 있을 수 있으므로 dets_raw를 순회
        det_raw = dets_raw[0]
        processed_det = {
            'scene_id': int(scene_id),
            'image_id': int(im_id),
            'score': det_raw.get('score', 1.0), # GT이므로 score는 1.0으로 설정
            'bbox': tuple(det_raw['bbox_est']), # GT bbox 키 (예: bbox_visib)
            'category_id': det_raw['obj_id'],
            'time': det_raw.get('time', 0.0)
        }

        # --- 📌 [수정] 올바른 mask_path 경로 및 RLE 변환 ---
        scene_folder_path = root_dir / dataset_name / "test" / brightness / f"{int(scene_id):06d}"
        mask_path = scene_folder_path / "mask_visib" / f"{int(im_id):06d}_000000.png"

        if os.path.exists(mask_path):
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_image is not None:
                binary_mask = (mask_image > 0).astype(np.uint8)
                rle = binary_mask_to_rle(binary_mask)
                processed_det['segmentation'] = rle
        
        processed_dets.append(processed_det)
        
        # 처리된 결과를 딕셔너리에 저장
        all_dets_per_image[new_key] = processed_dets
    # --- 📌 [수정] for 루프 종료 ---

    return (
        generate_test_list(all_dets_per_image),
        all_dets_per_image,
    )


def load_test_list_and_cnos_detections_est(
    root_dir, dataset_name, test_setting, max_det_per_object_id=None, 
    brightness="B50", rgb_sensor="AE", depth_sensor="0"
):
    bbox_base_path = Path("/ssd/sjkim/SAM-6D/Data/SS6D/outputs_whole_sam")
    if not bbox_base_path.exists():
        raise FileNotFoundError(f"Base detection path not found at: {bbox_base_path}")
    
    all_detections_list = []
    
    for obj_dir in os.listdir(bbox_base_path):
        json_path = bbox_base_path / obj_dir / brightness / f"depth_{depth_sensor}" / f"merged_ism_topscore_{rgb_sensor}.json"
        
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            detections_in_file = json.load(f)
            
            # --- 📌 [수정] 올바른 scene_id 할당 ---
            # "obj_2"에서 숫자 2를 추출하여 scene_id로 사용
            current_scene_id = int(obj_dir.split('_')[-1])
            for det in detections_in_file:
                det['scene_id'] = current_scene_id
            # ------------------------------------

            all_detections_list.extend(detections_in_file)

    # (이하 포맷 변환 로직은 동일)
    all_dets_per_image = {}
    for det_raw in all_detections_list:
        scene_id = det_raw['scene_id']
        im_id = det_raw['image_id']
        
        image_key = f"{scene_id:06d}_{im_id:06d}"
        
        if image_key not in all_dets_per_image:
            all_dets_per_image[image_key] = []
        
        processed_det = {
            'scene_id': scene_id,
            'image_id': im_id,
            'score': det_raw['score'],
            'bbox': tuple(det_raw['bbox']),
            'category_id': det_raw['category_id'],
            'time': det_raw.get('time', 0.0)
        }

        if 'segmentation' in det_raw:
            processed_det['segmentation'] = det_raw['segmentation']
        
        all_dets_per_image[image_key].append(processed_det)
        
    return (
        generate_test_list(all_dets_per_image),
        all_dets_per_image,
    )

def binary_mask_to_rle(mask):
    """0과 1로 구성된 이진 마스크를 RLE 형식으로 변환"""
    # pycocotools는 Fortran-style 메모리 레이아웃을 기대함
    mask_rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return mask_rle

































def load_test_list_and_init_locs(root_dir, dataset_name, init_loc_path, test_setting):
    # load init locs
    try:
        init_locs = load_bop_results(init_loc_path, additional_name="instance_id")
        instance_ids = [pose["instance_id"] for pose in init_locs]
        num_instances = len(np.unique(instance_ids))
        assert len(init_locs) % num_instances == 0
        num_hypothesis = int(len(init_locs) / num_instances)
    except:
        init_locs = load_bop_results(init_loc_path)
        num_hypothesis = 1
    # sort by image_id
    all_init_locs_per_image = group_by_image_level(init_locs, image_key="im_id")
    
    return (
            generate_test_list(all_init_locs_per_image),
            all_init_locs_per_image,
            num_hypothesis,
        )


if __name__ == "__main__":
    save_predictions_from_batched_predictions(
        "/home/nguyen/Documents/datasets/gigaPose_datasets/results/large_None/predictions/",
        dataset_name="icbin",
        model_name="large",
        run_id="12345678",
        is_refined=False,
    )
