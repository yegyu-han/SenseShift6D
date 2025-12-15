import os
import json
from tqdm import tqdm
import argparse

def merge_detections_in_sensor(sensor_dir, save_path, detection_type):
    merged = []

    for subdir in sorted(os.listdir(sensor_dir)):
        subdir_path = os.path.join(sensor_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        json_path = os.path.join(subdir_path, f"detection_{detection_type}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                try:
                    data = json.load(f)
                    merged.extend(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error in {json_path}: {e}")
        else:
            print(f"❌ No detection_ism.json found in {subdir_path}")

    # Save merged output
    with open(save_path, "w") as f:
        json.dump(merged, f)
    print(f"✅ Merged {len(merged)} detections into {save_path}")

def merge_detections_in_sensor_topscore(sensor_dir, save_path, detection_type, obj_id):
    merged = []

    for subdir in sorted(os.listdir(sensor_dir)):
        subdir_path = os.path.join(sensor_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        json_path = os.path.join(subdir_path, f"detection_{detection_type}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                try:
                    data = json.load(f)
                    if not data:
                        continue
                    # 가장 높은 score를 가진 detection 하나만 선택
                    best_det = max(data, key=lambda x: x.get("score", -1))
                    best_det["category_id"] = obj_id + 1
                    merged.append(best_det)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error in {json_path}: {e}")
        else:
            print(f"❌ No detection_{detection_type}.json found in {subdir_path}")

    # Save merged output
    with open(save_path, "w") as f:
        json.dump(merged, f)
    print(f"✅ Merged {len(merged)} best detections into {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge detection_ism.json files into a merged_detections.json per sensor")
    parser.add_argument("--obj_id", type=int, required=True, help="Object ID (e.g. 0)")
    # parser.add_argument("--depth_level", type=int, required=True, help="Depth level (e.g. 0)")
    parser.add_argument("--b_level", type=str, required=True, help="Brightness level (e.g. B25)")
    parser.add_argument("--base_path", type=str, default="/ssd/sjkim/SAM-6D/Data/SS6D", help="Base output directory")
    parser.add_argument("--gt_detection", type=bool, default=False, help="Used GT detection for pem?")
    parser.add_argument("--top1", type=bool, default=False, help="Use only top1 for sam?")
    
    args = parser.parse_args()

    if args.gt_detection:
        # base_path = os.path.join(args.base_path, "GT_ism")
        base_path = os.path.join(args.base_path, "outputs_whole_sam_gt")
    elif args.top1:
        base_path =  os.path.join(args.base_path, "outputs_whole_sam_top1")
    else:
        base_path =  os.path.join(args.base_path, "outputs_whole_sam")

    # Format paths
    obj_name = f"obj_{args.obj_id}"

    # Sensor list
    sensors = ["AE","AEG16", "AEG48", "AEG80", "AEG112",
               "E9G16", "E9G48", "E9G80", "E9G112",
               "E39G16", "E39G48", "E39G80", "E39G112",
               "E156G16", "E156G48", "E156G80", "E156G112",
               "E625G16", "E625G48", "E625G80", "E625G112",
               "E2500G16", "E2500G48", "E2500G80", "E2500G112"]

    # sensors = ["AEG16", "AEG48", "AEG80", "AEG112"]

    for depth in range(4):
        depth_name = f"depth_{depth}"
        base_dir = os.path.join(base_path, obj_name, args.b_level, depth_name)

        for sensor in tqdm(sensors, desc="Merging ism, pem detections"):
            sensor_dir = os.path.join(base_dir, sensor)
            # ism_save_path = os.path.join(base_dir, f"merged_ism_{sensor}.json")
            # merge_detections_in_sensor(sensor_dir, ism_save_path, "ism")
            # ism_save_path_topscore = os.path.join(base_dir, f"merged_ism_topscore_{sensor}.json")
            # merge_detections_in_sensor_topscore(sensor_dir, ism_save_path_topscore, "ism", obj_id=args.obj_id)

            # pem_save_path = os.path.join(base_dir, f"merged_pem_{sensor}.json")
            # merge_detections_in_sensor(sensor_dir, pem_save_path, "pem")
            pem_save_path_topscore = os.path.join(base_dir, f"merged_pem_topscore_{sensor}.json")
            merge_detections_in_sensor_topscore(sensor_dir, pem_save_path_topscore, "pem", obj_id=args.obj_id)
