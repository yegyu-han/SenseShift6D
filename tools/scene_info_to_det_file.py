import os
import os.path as osp
import mmcv
import json
from tqdm import tqdm

def save_json(path, content, sort=False):
    output_dir = os.path.dirname(path)
    os.makedirs(output_dir, exist_ok=True)

    with open(path, "w") as f:
        if isinstance(content, dict):
            f.write("{\n")
            if sort:
                content_sorted = sorted(content.items(), key=lambda x: x[0])
            else:
                content_sorted = content.items()
            for elem_id, (k, v) in enumerate(content_sorted):
                f.write('  "{}": {}'.format(k, json.dumps(v, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("}")
        elif isinstance(content, list):
            f.write("[\n")
            for elem_id, elem in enumerate(content):
                f.write("  {}".format(json.dumps(elem, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]")
        else:
            json.dump(content, f, sort_keys=True)

def main():
    base_scene_dir = "datasets/BOP_DATASETS/SenseShift6D/test/B50"
    output_path = "datasets/BOP_DATASETS/SenseShift6D/test/test_bboxes/scene_gt_info_bboxes.json"

    if os.path.exists(output_path):
        existing_data = mmcv.load(output_path)
    else:
        existing_data = {}

    outs = existing_data.copy()

    scene_dirs = sorted([d for d in os.listdir(base_scene_dir) if osp.isdir(osp.join(base_scene_dir, d))])

    for scene_dir in tqdm(scene_dirs):
        scene_id = int(scene_dir)
        gt_info_path = osp.join(base_scene_dir, scene_dir, "scene_gt_info.json")

        if not osp.exists(gt_info_path):
            continue

        ds = mmcv.load(gt_info_path)

        for image_id, infos in ds.items():
            scene_im_id = f"{scene_id}/{image_id}"

            for info in infos:
                obj_id = scene_id+1  # 필요한 경우 info["obj_id"]로 변경 가능
                score = 1.0
                bbox = info["bbox_obj"]
                time = 1

                cur_dict = {
                    "bbox_est": bbox,
                    "obj_id": obj_id,
                    "score": score,
                    "time": time,
                }

                if scene_im_id in outs:
                    outs[scene_im_id].append(cur_dict)
                else:
                    outs[scene_im_id] = [cur_dict]

    save_json(output_path, outs)
    print(f"Saved combined bbox json to {output_path}")

if __name__ == "__main__":
    main()
