import csv
import argparse
import numpy as np
from sklearn.neighbors import KDTree
# from lib.utils import compute_add_score, compute_adds_score
# import glob
# import pdb
import json

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path: sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add project root
root_path = osp.abspath(osp.join(this_dir, '..'))
add_path(root_path)

def compute_add_score_with_auc(pts3d, diameter, pose_gt, pose_pred, num_thresholds=100):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred
    pts_xformed_gt = R_gt @ pts3d.T + t_gt
    pts_xformed_pred = R_pred @ pts3d.T + t_pred

    distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
    mean_dist = np.mean(distance)

    if np.isnan(mean_dist):
        print(f"[Warning] NaN detected in ADD distance!")

    # AUC 계산
    thresholds = np.linspace(0, 0.10, num=num_thresholds)
    sum_correct = sum(mean_dist < diameter * t for t in thresholds)
    auc_score = sum_correct / num_thresholds

    return auc_score, mean_dist


def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.05):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred
    pts_xformed_gt = R_gt * pts3d.transpose() + t_gt
    pts_xformed_pred = R_pred * pts3d.transpose() + t_pred

    distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
    mean_dist = np.mean(distance)
    if np.isnan(mean_dist):
        print(f"[Warning] NaN detected in ADD distance!")

    threshold = diameter * percentage
    passed = (mean_dist < threshold)
    # print(f"ADD distance: {mean_dist:.4f} m")
    return passed, mean_dist

def compute_adds_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.05):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred

    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            mean_distances[i] = np.inf
            continue
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
        distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
        mean_distances[i] = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_id', type=int, default='0', help='Object ID (e.g. 0 for spray, 1 for pringles')
    parser.add_argument('--b_level', type=str, default='B5', help='Brightness level (e.g. B25)')
    parser.add_argument('--base_path', type=str, default='/ssd/sjkim/SAM-6D/Data/SS6D', help='Base output directory')
    parser.add_argument('--gt_old', type=bool, default=False, help='use gt_old or not for obj2 (tincase)')
    # parser.add_argument('--prediction_file', type=str )
    args = parser.parse_args()
    args.cad_id= args.object_id + 1  # Convert to 1-based index for CAD models
    return args

def read_3d_points():
    filename = f'/ssd/sjkim/SAM-6D/Data/SS6D/models/obj_{args.cad_id:06d}.ply'
    with open(filename) as f:
        in_vertex_list = False
        vertices = []
        in_mm = False
        for line in f:
            if in_vertex_list:
                vertex = line.split()[:3]
                vertex = np.array([float(vertex[0]),
                                   float(vertex[1]),
                                   float(vertex[2])], dtype=np.float32)
                if in_mm:
                    vertex = vertex / np.float32(10) # mm -> cm
                vertex = vertex / np.float32(100)
                vertices.append(vertex)
                if len(vertices) >= vertex_count:
                    break
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('end_header'):
                in_vertex_list = True
            elif line.startswith('element face'):
                in_mm = True
    return np.matrix(vertices)

def read_diameter():

    filename = '/ssd/sjkim/SAM-6D/Data/SS6D/models/models_info.json'
    with open(filename, "r") as f:
       json_data = json.load(f)
    diameter_m = json_data[str(args.cad_id)]['diameter'] / 1000.0  # Convert mm to m
    # print(f"Diameter: {diameter_m:.4f} m, threshold: {diameter_m * 0.1:.4f} m")
    return diameter_m   # ✅ mm → m (일관성 있게)

def load_gt_poses(gt_file):
    with open(gt_file, 'r') as f:
        gt_json = json.load(f)
    gt_poses = {}
    for img_id_str, annotations in gt_json.items():
        img_id = int(img_id_str)
        pose = annotations[0]  # Assume only one object per image
        R = np.array(pose['cam_R_m2c']).reshape(3, 3)  # 3x3 rotation matrix
        t = np.array(pose['cam_t_m2c']).reshape(3, 1) / 1000  # mm -> m
        gt_poses[img_id] = (R, t)
    return gt_poses

if __name__ == '__main__':
    args = parse_args()
    depth_levels = [0, 1, 2, 3]  # Assuming depth level is fixed for this script
    # sensor = 'AE'
    sensors = ["AE",
                "E9G16", "E9G48", "E9G80", "E9G112",
                "E39G16", "E39G48", "E39G80", "E39G112", 
                "E156G16", "E156G48", "E156G80", "E156G112", 
                "E625G16", "E625G48", "E625G80", "E625G112", 
                "E2500G16", "E2500G48", "E2500G80", "E2500G112"]

    results = []  # 결과를 저장할 리스트

    for sensor in sensors:
        for depth_level in depth_levels:

            prediction_file = osp.join(
                args.base_path,
                "outputs_whole_sam", 
                f"obj_{args.object_id}", 
                args.b_level, 
                f"depth_{depth_level}", 
                f"merged_pem_topscore_{sensor}.json")

            # 예측 파일 로드 (list of dicts)
            with open(prediction_file, 'r') as f:
                records = json.load(f)
            record_dict = {item['image_id']: item for item in records}

            # GT pose 불러오기
            gt_file = f"/ssd/sjkim/SAM-6D/Data/SS6D/test/{args.b_level}/{args.object_id:06d}/scene_gt.json"
            gt_file_old = f"/ssd/sjkim/SAM-6D/Data/SS6D/test/{args.b_level}/{args.object_id:06d}/scene_gt_old.json"
            if args.object_id == 2 and args.gt_old == True:
                gt_poses = load_gt_poses(gt_file_old)
            else:
                gt_poses = load_gt_poses(gt_file)

            # 3D 모델 및 diameter
            pts3d = read_3d_points()
            diameter = read_diameter()

            compute_score = compute_add_score  # or compute_adds_score for symmetric

            # 평가 루프
            image_count = len(gt_poses.keys())
            AUC_SCORE = 0
            for image_id in sorted(gt_poses.keys()):
                # detection 실패하면 0
                if image_id not in record_dict:
                    print(f"[Warning] Prediction missing for image_id {image_id}")
                    auc_score = 0.0
                    adx=10000
                else:
                    R_gt, t_gt = gt_poses[image_id]
                    R_pred = np.array(record_dict[image_id]['R'])
                    t_pred = np.array(record_dict[image_id]['t']).reshape(3, 1) / 1000.0  # mm -> m
                    
                    # print(f"[Image {image_id}] ADD distance: {adx:.4f} m")

                    auc_score, adx = compute_add_score_with_auc(
                        pts3d,
                        diameter,
                        (R_gt, t_gt),
                        (R_pred, t_pred)
                    )

                # 결과 리스트에 저장
                results.append({
                    "sensor": sensor,
                    "depth_level": depth_level,
                    "image_id": image_id,
                    "add_distance": round(adx, 6),
                    "auc_score": round(auc_score, 4)
                })

                AUC_SCORE += auc_score

            print(f"👾 {sensor}-{depth_level}: AUC score = {AUC_SCORE / image_count:.4f}")

        # 📄 이미지별 결과 CSV 저장
        if args.object_id == 2 and args.gt_old == True:
            per_image_csv_path = osp.join(args.base_path, "ADD_pem_results_sam", f"auc_per_image_obj{args.object_id}_{args.b_level}_old.csv")
        else:
            per_image_csv_path = osp.join(args.base_path, "ADD_pem_results_sam", f"auc_per_image_obj{args.object_id}_{args.b_level}.csv")
        with open(per_image_csv_path, mode='w', newline='') as csvfile:
            fieldnames = ['sensor', 'depth_level', 'image_id', 'add_distance', 'auc_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)



    # 📄 센서별 요약 CSV 저장
    # summary_csv_path = osp.join(args.base_path, "ADD_pem_results_sam", f"add_summary_obj{args.object_id}_{args.b_level}.csv")
    # with open(summary_csv_path, mode='w', newline='') as csvfile:
    #     fieldnames = ['sensor', 'depth_level', 'ADD_score', 'passed_count', 'total_images']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
        
    #     from collections import defaultdict
    #     score_map = defaultdict(lambda: {'passed': 0, 'total': 0})
        
    #     for row in results:
    #         key = (row['sensor'], row['depth_level'])
    #         score_map[key]['passed'] += row['passed']
    #         score_map[key]['total'] += 1

    #     for (sensor, depth_level), score in score_map.items():
    #         passed = score['passed']
    #         total = score['total']
    #         writer.writerow({
    #             'sensor': sensor,
    #             'depth_level': depth_level,
    #             'ADD_score': round(passed / total, 4),
    #             'passed_count': passed,
    #             'total_images': total
    #         })

    # print(f"📊 센서별 ADD score 요약 저장 완료: {summary_csv_path}")

    print("✅ 모든 작업 완료!")

