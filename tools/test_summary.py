import os, sys, json
import pandas as pd
import numpy as np
sys.path.append("/hdd/tgyoon/HiPose/HiPose/hipose/bop_toolkit")
from bop_toolkit_lib import inout, pose_error, renderer

metric = "auc"
brightness_list = [
                   'B5',
                   'B25',
                   'B50',
                   'B75',
                   'B100'
                   ]
rgb_sensor_list = [
                   'AE',
                   'E9G16', 'E9G48', 'E9G80', 'E9G112',
                   'E39G16', 'E39G48', 'E39G80', 'E39G112',
                   'E156G16', 'E156G48', 'E156G80', 'E156G112',
                   'E625G16', 'E625G48', 'E625G80', 'E625G112',
                   'E2500G16', 'E2500G48', 'E2500G80', 'E2500G112']
depth_list = ["0", "1", "2", "3"]
obj_list = ["spray", "pringles", "tincase", "sandwich", "mouse"]
obj_id_list = [1, 2, 3, 4, 5]
obj_diameter = {"1":277.8929629556948,
                "2":121.84687732405808,
                "3":130.9916633062778,
                "4":164.41491007818098,
                "5":106.26554674838228}

def Calculate_Pose_Error_Main(R_GT,t_GT, R_predict, t_predict, vertices):
    t_GT = t_GT.reshape((3,1))
    t_predict = np.array(t_predict).reshape((3,1))

    return pose_error.add(R_predict, t_predict, R_GT, t_GT, vertices)

if __name__ == "__main__":
    # base info
    width, height = 1280, 720
    identity_transform = {
        'R': np.identity(3),
        't': np.zeros((3, 1))
    }
    vsd_delta = 15
    vsd_taus = list(np.arange(0.05, 0.51, 0.05))
    vsd_th = [th for th in np.arange(0.05, 0.51, 0.05)]
    mssd_th = [th for th in np.arange(0.05, 0.51, 0.05)]
    mspd_th = [th for th in np.arange(5, 51, 5)]
    
    ### modified for each model to combine all experimental results. ###
    file_name = "bop_summary_gigapose.csv"
    if not os.path.exists(file_name):
        list_of_dfs = []
        for brightness in brightness_list:
            for rgb_sensor in rgb_sensor_list:
                for depth in depth_list:
                    if brightness == 'B5':
                        df_path = f"/hdd/tgyoon/GiGaPose/gigapose/gigaPose_datasets/results/large_ss6d_{brightness}_{rgb_sensor}_{depth}/refined_multiple_predictions/large-pbrreal-rgb-mmodel_SENSESHIFT6D-test_0MultiHypothesisMultiHypothesis.csv"
                    else:
                        df_path = f"/hdd/tgyoon/GiGaPose/gigapose/gigaPose_datasets/results/large_ss6d_{brightness}_{rgb_sensor}_{depth}/refined_multiple_predictions/large-pbrreal-rgb-mmodel_SENSESHIFT6D-test_{depth}MultiHypothesisMultiHypothesis.csv"
                    try:
                        df = pd.read_csv(df_path)
                    except FileNotFoundError:
                        print(f"Does file exist? {os.path.exists(df_path)}")
                        continue
                    list_of_dfs.append(df)
        all_results_df = pd.concat(list_of_dfs, ignore_index=True)
    ####################################################################

        final_dfs = []
        bop_renderer = renderer.create_renderer(width, height, renderer_type='python')
        for obj_id in obj_id_list:
            results_df = all_results_df[all_results_df["obj_id"] == obj_id].copy()
            gt_json_path = f"/hdd/tgyoon/GiGaPose/gigapose/gigaPose_datasets/datasets/SENSESHIFT6D/test/B5/{obj_id-1:06d}/scene_gt.json"
            camera_json_path = f"/hdd/tgyoon/GiGaPose/gigapose/gigaPose_datasets/datasets/SENSESHIFT6D/test/B5/{obj_id-1:06d}/scene_camera.json"
            mesh_path = f"/hdd/tgyoon/GiGaPose/gigapose/gigaPose_datasets/datasets/SENSESHIFT6D/models/obj_{obj_id:06d}.ply"

            with open(camera_json_path, "r") as f:
                camera_data = json.load(f)
            with open(gt_json_path, "r") as f:
                gt_data = json.load(f)
            
            bop_renderer.add_object(obj_id, mesh_path)
            vertices = inout.load_ply(mesh_path)["pts"]

            add_errors = []
            auc_errors = []
            vsd_errors = []
            mssd_errors = []
            mspd_errors = []
            for row in results_df.itertuples():
                R_predict = np.array(row.R.split()).astype(float).reshape(3, 3)
                t_predict = np.array(row.t.split()).astype(float).reshape(3, 1)
                depth_path = f"/hdd/tgyoon/GiGaPose/gigapose/gigaPose_datasets/datasets/SENSESHIFT6D/test/{row.brightness}/{obj_id-1:06d}/depth/{row.depth_sensor}/{row.im_id:06d}.png"
                depth_im = inout.load_depth(depth_path)
                K = np.array(camera_data[str(row.im_id)]["cam_K"]).reshape(3, 3)
                gt_info = gt_data[str(row.im_id)][0]
                r_GT = np.array(gt_info["cam_R_m2c"]).reshape(3, 3)
                t_GT = np.array(gt_info["cam_t_m2c"]).reshape(3, 1)
                add = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                vsd = pose_error.vsd(R_predict, t_predict, r_GT, t_GT, depth_im, K, vsd_delta, vsd_taus, True, obj_diameter[str(obj_id)], bop_renderer, obj_id, 'step')
                vsd_re = []
                for th in vsd_th:
                    for e in vsd:
                        vsd_re.append(float(e < th))
                vsd_avg = np.mean(vsd_re)
                mssd = pose_error.mssd(R_predict, t_predict, r_GT, t_GT, vertices, [identity_transform])
                mssd_re = []
                for th in mssd_th:
                    mssd_re.append(float(mssd < th * obj_diameter[str(obj_id)]))
                mssd_avg = np.mean(mssd_re)
                mspd = pose_error.mspd(R_predict, t_predict, r_GT, t_GT, K, vertices, [identity_transform])
                mspd_re = []
                factor = 640.0 / width
                for th in mspd_th:
                    mspd_re.append(float(factor * mspd < th))
                mspd_avg = np.mean(mspd_re)
                th = np.linspace(0, 0.10, num=100)
                sum_correct = 0
                for t in th:
                    if add < obj_diameter[str(obj_id)] * t:
                        sum_correct = sum_correct + 1
                auc = sum_correct/100
                add_errors.append(add)
                auc_errors.append(auc)
                vsd_errors.append(vsd_avg)
                mssd_errors.append(mssd_avg)
                mspd_errors.append(mspd_avg)
                print(row.scene_id, row.im_id, row.brightness, row.rgb_sensor, row.depth_sensor, row.obj_id, add, vsd_avg, mssd_avg, mspd_avg)

            results_df["add"] = add_errors
            results_df["auc"] = auc_errors
            results_df["vsd"] = vsd_errors
            results_df["mssd"] = mssd_errors
            results_df["mspd"] = mspd_errors
            final_dfs.append(results_df)

        all_results_df = pd.concat(final_dfs, ignore_index=True)
        all_results_df.to_csv(file_name)
    else:
        all_results_df = pd.read_csv(file_name)

    exclude_list = ['AEG16', 'AEG48', 'AEG80', 'AEG112']
    all_results_df = all_results_df[~all_results_df['rgb_sensor'].isin(exclude_list)]

    grouping_cols = ['scene_id', 'im_id', 'obj_id', 'brightness']
    all_unique_keys = all_results_df[grouping_cols].drop_duplicates().reset_index(drop=True)

    depth_only_scope_df = all_results_df[all_results_df['rgb_sensor'] == 'AE'].copy()
    depth_idx = depth_only_scope_df.groupby(grouping_cols)[metric].idxmax()
    depth_oracle_result = depth_only_scope_df.loc[depth_idx]

    rgb_only_scope_df = all_results_df[(all_results_df['depth_sensor'] == 0) & (all_results_df['rgb_sensor'] != 'AE')].copy()
    rgb_idx = rgb_only_scope_df.groupby(grouping_cols)[metric].idxmax()
    rgb_oracle_result = rgb_only_scope_df.loc[rgb_idx]

    cols_to_merge = grouping_cols + [metric, 'rgb_sensor', 'depth_sensor']

    depth_df = pd.merge(all_unique_keys, depth_oracle_result[cols_to_merge], on=grouping_cols, how='left')

    rgb_df = pd.merge(all_unique_keys, rgb_oracle_result[cols_to_merge], on=grouping_cols, how='left')

    depth_df[metric] = depth_df[metric].fillna(0)
    rgb_df[metric] = rgb_df[metric].fillna(0)

    base_df = all_results_df[(all_results_df['rgb_sensor'] == 'AE') & (all_results_df['depth_sensor'] == 0)].copy()

    dynamic_scope_df = all_results_df[(all_results_df['rgb_sensor'] != 'AE')].copy()
    dynamic_idx = dynamic_scope_df.groupby(['scene_id', 'im_id', 'obj_id', 'brightness'])[metric].idxmax()
    dynamic_df = all_results_df.loc[dynamic_idx].copy()

    results = []
    for obj_id in obj_id_list:
        base = base_df[(base_df['obj_id'] == obj_id)][metric].mean()
        depth = depth_df[(depth_df['obj_id'] == obj_id)][metric].mean()
        rgb = rgb_df[(rgb_df['obj_id'] == obj_id)][metric].mean()
        fixed_scope_df = all_results_df[(all_results_df['rgb_sensor'] != 'AE') & (all_results_df['obj_id'] == obj_id)].copy()
        group_means = fixed_scope_df.groupby(['rgb_sensor', 'depth_sensor'])[metric].mean()
        best_group_name = group_means.idxmax() # 예: ('AE', '0')
        best_rgb_sensor, best_depth_sensor = best_group_name
        fixed_df = fixed_scope_df[
            (fixed_scope_df['rgb_sensor'] == best_rgb_sensor) &
            (fixed_scope_df['depth_sensor'] == best_depth_sensor)
        ].copy()
        fixed = fixed_df[(fixed_df['obj_id'] == obj_id)][metric].mean()
        dynamic = dynamic_df[(dynamic_df['obj_id'] == obj_id)][metric].mean()
        rand = all_results_df[(all_results_df['obj_id'] == obj_id)][metric].mean()
        
        results.append({
            "Object": obj_list[obj_id-1],
            "Baseline": base * 100,
            "Depth-Only": depth * 100,
            "ΔDepth": (depth - base) * 100,
            "RGB-Only": rgb * 100,
            "ΔRGB": (rgb - base) * 100,
            "Oracle-Fixed": fixed * 100,
            "ΔFixed": (fixed - base) * 100,
            "Oracle-Dynamic": dynamic * 100,
            "ΔDynamic": (dynamic - base) * 100,
            "Rand": rand * 100,
        })

    df_result = pd.DataFrame(results)

    overall = {
        "Object": "Overall",
        "Baseline": df_result["Baseline"].mean(),
        "Depth-Only": df_result["Depth-Only"].mean(),
        "ΔDepth": df_result["ΔDepth"].mean(),
        "RGB-Only": df_result["RGB-Only"].mean(),
        "ΔRGB": df_result["ΔRGB"].mean(),
        "Oracle-Fixed": df_result["Oracle-Fixed"].mean(),
        "ΔFixed": df_result["ΔFixed"].mean(),
        "Oracle-Dynamic": df_result["Oracle-Dynamic"].mean(),
        "ΔDynamic": df_result["ΔDynamic"].mean(),
        "Rand": df_result["Rand"].mean(),
    }

    df_result = pd.concat([df_result, pd.DataFrame([overall])], ignore_index=True)

    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(df_result[[
            "Object",
            "Baseline",
            "Depth-Only", "ΔDepth",
            "RGB-Only", "ΔRGB",
        ]])
        print(df_result[[
            "Oracle-Fixed", "ΔFixed",
            "Oracle-Dynamic", "ΔDynamic",
            "Rand"
        ]])
