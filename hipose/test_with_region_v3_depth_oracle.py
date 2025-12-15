import os
import sys
import glob

sys.path.insert(0, os.getcwd())
sys.path.append("../bop_toolkit")

import argparse
from tqdm import tqdm
import torch
import numpy as np

from bop_toolkit_lib import inout

from config_parser import parse_cfg
from tools_for_BOP.common_dataset_info import get_obj_info
from tools_for_BOP import bop_io, write_to_cvs, ss6d_io_depth_oracle
from binary_code_helper.CNN_output_to_pose_region import load_dict_class_id_3D_points

from binary_code_helper.CNN_output_to_pose_region_for_test_with_region_v3 import CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v7

from binary_code_helper.generate_new_dict import generate_new_corres_dict_and_region
from models.ffb6d import FFB6D
from models.common import ConfigRandLA
from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP
from get_detection_results import ycbv_select_keyframe


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def compute_auc_posecnn(errors):
    # NOTE: Adapted from https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m
    errors = errors.copy()
    d = np.sort(errors)
    d[d > 0.1] = np.inf
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids-1]) * mpre[ids]).sum() * 10
    return ap


def te(t_est, t_gt):
    """Translational Error.

    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :return: The calculated error.
    """
    t_est = t_est.flatten()
    t_gt = t_gt.flatten()
    assert t_est.size == t_gt.size == 3
    error = np.linalg.norm(t_gt - t_est)
    return error

def re(R_est, R_gt):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))

    return rd_deg


def main(configs):
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder=configs['training_data_folder']
    training_data_folder_2=configs['training_data_folder_2']
    test_folder=configs['test_folder']                                  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']              # the percentage of second dataset in the batch
    num_workers = configs['num_workers']
    train_obj_visible_theshold = configs['train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values, usually 0.2
    #### network settings
    convnext = configs.get('convnext', False)
    if convnext is False:
        from bop_dataset_3d import bop_dataset_single_obj_3d, ss6d_dataset_single_obj_3d
    elif convnext.startswith('convnext'):
        from bop_dataset_3d_convnext_backbone import bop_dataset_single_obj_3d, ss6d_dataset_single_obj_3d
    fusion = configs.get('fusion', False)
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']              # now only support "L1" or "BCE"          

    #### augmentations
    Detection_reaults=configs['Detection_reaults']                       # for the test, the detected bounding box provided by GDR Net
    padding_ratio=configs['padding_ratio']                               # pad the bounding box for training and test
    resize_method = configs['resize_method']                             # how to resize the roi images to 256*256

    # augmentation
    use_peper_salt= configs['use_peper_salt']
    use_motion_blur= configs['use_motion_blur']
    interpolate_depth = configs['interpolate_depth']  
    shift_center = configs['shift_center']  
    aug_depth = configs['aug_depth']
    aug_depth_megapose6d = configs.get('aug_depth_megapose6d', False)
    assert not (aug_depth & aug_depth_megapose6d)

    # pixel code settings
    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']

    # ablation study
    region_bit = configs['region_bit']
    threshold = configs['threshold']

    # additional settings
    config_file_name = configs['checkpoint_file'].split("/")[8]
    general = configs['general']
    brightness = configs['brightness']

    torch.manual_seed(0)      # the both are only good for ablation study
    np.random.seed(0)         # if can be removed in the final experiments

    calc_add_and_adi=True

    # get dataset informations
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io.get_dataset(bop_path,dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1)
    if obj_name in symmetry_obj:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
        main_metric_name = 'ADI'
        supp_metric_name = 'ADD'
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
        main_metric_name = 'ADD'
        supp_metric_name = 'ADI'
    
    mesh_path = model_plys[obj_id+1] # mesh_path is a dict, the obj_id should start from 1
    obj_diameter = model_info[str(obj_id+1)]['diameter']
    print("obj_diameter", obj_diameter)
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id+1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(divide_number_each_itration)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(number_of_itration)
   
    GT_code_infos = [divide_number_each_itration, number_of_itration, total_numer_class]

    vertices = inout.load_ply(mesh_path)["pts"]

    sensor_list = ['AE', 'AEG16', 'AEG48', 'AEG80', 'AEG112',
                'E9G16', 'E9G48', 'E9G80', 'E9G112',
                'E39G16', 'E39G48', 'E39G80', 'E39G112',
                'E156G16', 'E156G48', 'E156G80', 'E156G112',
                'E625G16', 'E625G48', 'E625G80', 'E625G112',
                'E2500G16', 'E2500G48', 'E2500G80', 'E2500G112']

    ae_list = ['AE', 'AEG16', 'AEG48', 'AEG80', 'AEG112']

    depth_preset_list = ['0','1', '2', '3'] 

    # path = os.path.join(eval_output_path, "ADD_result", config_file_name)
    setting = configs['checkpoint_file'].split('/')[-2].split('_')[-2]
    path = eval_output_path + config_file_name
    print("path: ", path)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, "{}_{}_{}_{}_{}.txt".format(dataset_name, obj_name, brightness, setting, obj_name))
    if os.path.exists(path):
        os.remove(path)

    # success_img_ids = set()
    success_ad5_img_ids = set()
    AD5_sensor_mean = []
    sample_path = os.path.join(bop_path, dataset_name, test_folder, brightness, str(obj_id).zfill(6), 'rgb/AE')
    n_samples = len(glob.glob(os.path.join(sample_path, '*.png')))
    ADD_min_error = np.ones(n_samples) * 10000
    ADD_min_global = np.full(n_samples, np.inf) # 수정, 전체‑oracle (depth + sensor 모두 포함) – 이미지별 최소값
    preset_min_errors = {dp: np.full(n_samples, np.inf) for dp in depth_preset_list} # 수정, 프리셋마다 이미지별 최소값
    RE_min_error = np.ones(n_samples) * 10000
    TE_min_error = np.ones(n_samples) * 10000
    best_rgb = np.full(n_samples, '', dtype=object)
    best_depth = np.full(n_samples, '', dtype=object)

    depth_ad5_passed   = {dp: np.zeros(n_samples) for dp in depth_preset_list}
    success_ad5_by_dp  = {dp: set()              for dp in depth_preset_list}

    non_ae_add_means, non_ae_r005 = [], []
    ae_add_means, ae_r005 = [], []
    per_depth_non_ae = {dp: {'add':[], 'r005':[]} for dp in depth_preset_list}
    per_depth_ae = {dp: {'add':[], 'r005':[]} for dp in depth_preset_list}

    per_depth_auc_non_ae = {dp: [float('inf')] for dp in depth_preset_list}  # non-AE AUC 저장
    per_depth_auc_ae = {dp: [float('inf')] for dp in depth_preset_list}      # AE AUC 저장
    global_auc_non_ae_max = -np.inf  # non-AE 글로벌 AUC의 최댓값 추적
    global_auc_ae_max = -np.inf   
    global_auc_all_max = -np.inf

    for d_preset in depth_preset_list: # 수정
        print("d_preset: ", d_preset)
        for sensor in sensor_list:

            cvs_path = os.path.join(eval_output_path, brightness, config_file_name)
            if not os.path.exists(cvs_path):
                os.makedirs(cvs_path)
            cvs_path = os.path.join(cvs_path, "{}_{}_{}_{}_{}".format(dataset_name, obj_name, brightness, d_preset, sensor))

            # define test data loader
            if not bop_challange:
                dataset_dir_test,_,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = ss6d_io_depth_oracle.get_dataset(bop_path, dataset_name,train=False, data_folder=test_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold, \
                general=general, brightness=brightness, sensor=sensor, depth_preset_choice=d_preset ) # 수정
            else:
                print("use BOP test images")
                dataset_dir_test,_,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_bop_challange_test_data(bop_path, dataset_name, target_obj_id=obj_id+1, data_folder=test_folder)
    
            
            has_gt = True
            if test_gts[obj_id][0] == None:
                has_gt = False

            if Detection_reaults != 'none':
                if configs['detector']=='FCOS':
                    from get_detection_results import get_detection_results, get_detection_scores
                elif configs['detector']=='MASKRCNN':
                    from get_mask_rcnn_results import get_detection_results, get_detection_scores
                Det_Bbox = get_detection_results(Detection_reaults, test_rgb_files[obj_id], obj_id+1, 0)
                scores = get_detection_scores(Detection_reaults, test_rgb_files[obj_id], obj_id+1, 0)
            else:
                Det_Bbox = None

            test_dataset = ss6d_dataset_single_obj_3d(
                                                    dataset_dir_test, test_folder, test_rgb_files[obj_id], test_depth_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id], 
                                                    test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False, BoundingBox_CropSize_image,
                                                    GT_code_infos, 
                                                    padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox=Det_Bbox,
                                                    use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur, dict_class_id_3D_points=dict_class_id_3D_points,
                                                    interpolate_depth=interpolate_depth, 
                                                    aug_depth=aug_depth,
                                                    aug_depth_megapose6d=aug_depth_megapose6d,
                                                    shift_center=shift_center
                                                )

            print("test image example:", test_rgb_files[obj_id][0], flush=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

            binary_code_length = number_of_itration
            print("predicted binary_code_length", binary_code_length)
            configs['binary_code_length'] = binary_code_length
        
            rndla_cfg = ConfigRandLA
            net = FFB6D(
                n_classes=1, n_pts=480 * 640 // 24 , rndla_cfg=rndla_cfg,
                number_of_outputs=binary_code_length + 1, fusion=fusion, convnext=convnext
            )

            if torch.cuda.is_available():
                net=net.cuda()

            checkpoint = torch.load( configs['checkpoint_file'] )
            net.load_state_dict(checkpoint['model_state_dict'])

            net.eval()
            #test with test data
            AD5_passed=np.zeros(len(test_loader.dataset))
            RE_error=np.zeros(len(test_loader.dataset))
            TE_error=np.zeros(len(test_loader.dataset))
            ADX_error=np.zeros(len(test_loader.dataset))
            AUC_ADX_error=np.zeros(len(test_loader.dataset))


            print("test dataset")
            print(len(test_loader.dataset))

            bit2class_id_center_and_region = {}
            for bit in range(region_bit + 1, 17):
                bit2class_id_center_and_region[bit] = generate_new_corres_dict_and_region(dict_class_id_3D_points, 16, bit)

            # complete bit2class_id_center_and_region so that all regions share the same shape, default: 32
            region_max_points = pow(2, 15 - region_bit)
            for bit in range(region_bit + 1, 17):
                for center_and_region in bit2class_id_center_and_region[bit].values():
                    region = center_and_region['region']
                    assert region.shape[0] <= region_max_points
                    if region.shape[0] < region_max_points:
                        region_new = np.zeros([region_max_points, 3])
                        region_new[:region.shape[0]] = region
                        region_new[region.shape[0]:] = region[0]
                        center_and_region['region'] = region_new

            img_ids = []
            scene_ids = []
            estimated_Rs = []
            estimated_Ts = []
            for rgb_fn in test_rgb_files[obj_id]:
                rgb_fn = rgb_fn.split("/")
                scene_id = rgb_fn[-4]
                img_id = rgb_fn[-1].split(".")[0]
                img_ids.append(img_id)
                scene_ids.append(scene_id)

            for batch_idx, (inputs, targets, rgb_fns) in enumerate(tqdm(test_loader)):
                # do the prediction and get the predicted binary code
                if not inputs: # no valid detected bbox
                    R_ = np.zeros((3,3))
                    R_[0,0] = 1
                    R_[1,1] = 1
                    R_[2,2] = 1
                    estimated_Rs.append(R_)
                    estimated_Ts.append(np.zeros((3,1)))
                    continue
                if torch.cuda.is_available():
                    for key in inputs:
                        inputs[key] = inputs[key].cuda()
            
                pred_masks_out, pred_codes_out = net(inputs)  # [bsz, 1, 2730], [bsz, 16, 2730]
                pred_masks_probability = torch.sigmoid(pred_masks_out).detach().cpu().numpy()
                pred_codes_probability = torch.sigmoid(pred_codes_out).detach().cpu().numpy()

                targets['Rs'] = targets['Rs'].detach().cpu().numpy()
                targets['ts'] = targets['ts'].detach().cpu().numpy()

                ################ code for analysis the error in code prediction
                class_code_target = targets['obj_pts_code'].detach().cpu().numpy().transpose(0, 2, 3, 1)
                mask_target = targets['obj_pts_mask'].detach().cpu().numpy()
                cld_xyz0 = inputs['cld_xyz0'].detach().cpu().numpy()
                for counter, (r_GT, t_GT, rgb_fn) in enumerate(zip(targets['Rs'] , targets['ts'], rgb_fns )):
                    R_predict, t_predict, success = CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v7(
                        cld_xyz0[counter],pred_masks_probability[counter], 
                        pred_codes_probability[counter],
                        bit2class_id_center_and_region=bit2class_id_center_and_region,
                        dict_class_id_3D_points=dict_class_id_3D_points,
                        region_bit=region_bit,
                        threshold=threshold,
                        uncertain_threshold=0.02,
                        mean=configs['mean'],
                        mask_target=mask_target[counter],
                        class_code_image_target=class_code_target[counter], 
                        R_gt=r_GT, 
                        t_gt=t_GT,
                        rgb =inputs['rgb'][counter],
                        depth = inputs['dpt_map_m'][counter],
                        debug_image_dir = os.path.abspath(
                                os.path.join(eval_output_path, scene_ids[batch_idx], img_ids[batch_idx])),
                        ) 

                    if success:     
                        if shift_center:
                            t_predict = t_predict + 1000. * inputs['original_center'].detach().cpu().numpy()[counter].reshape((3,1))
                        estimated_Rs.append(R_predict)
                        estimated_Ts.append(t_predict)
                    else:
                        R_ = np.zeros((3,3))
                        R_[0,0] = 1
                        R_[1,1] = 1
                        R_[2,2] = 1
                        estimated_Rs.append(R_)
                        estimated_Ts.append(np.zeros((3,1)))

                    adx_error = 10000
                    re_error = 10000
                    te_error = 10000
                    if success and has_gt:
                        adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                        re_error = re(r_GT, R_predict)
                        te_error = te(t_GT, t_predict)
                        if np.isnan(adx_error): adx_error = 10000
                        if np.isnan(re_error): re_error = 10000
                        if np.isnan(te_error): te_error = 10000

                    if adx_error < obj_diameter*0.05:
                        AD5_passed[batch_idx] = 1
                        if sensor not in ae_list:
                            success_ad5_img_ids.add(rgb_fn.split("/")[-1][:6])
                            depth_ad5_passed[d_preset][batch_idx] = 1   
                            success_ad5_by_dp[d_preset].add(rgb_fn.split("/")[-1][:6])  

                    ADX_error[batch_idx] = adx_error
                    RE_error[batch_idx] = re_error
                    TE_error[batch_idx] = te_error

                    if adx_error < ADD_min_error[batch_idx] and sensor not in ae_list:
                        ADD_min_error[batch_idx] = adx_error
                        RE_min_error[batch_idx] = re_error
                        TE_min_error[batch_idx] = te_error
                        best_rgb[batch_idx] = rgb_fn
                        best_depth[batch_idx] = d_preset

                    # 수정, preset별 오라클 업뎃
                    if adx_error < preset_min_errors[d_preset][batch_idx]:
                        if sensor not in ae_list:
                            preset_min_errors[d_preset][batch_idx] = adx_error
                    
                    # 수정, 전체 오라클 업뎃
                    if adx_error < ADD_min_global[batch_idx]:
                        if sensor not in ae_list:
                            ADD_min_global[batch_idx] = adx_error

                    th = np.linspace(0, 0.10, num=100)
                    sum_correct = 0
                    for t in th:
                        if adx_error < obj_diameter*t:
                            sum_correct = sum_correct + 1
                    AUC_ADX_error[batch_idx] = sum_correct/100

                    if sensor not in ae_list:
                        if AUC_ADX_error[batch_idx] < per_depth_auc_non_ae[d_preset]:
                            per_depth_auc_non_ae[d_preset] = AUC_ADX_error[batch_idx]  # 최소값 갱신
                    else:
                        if AUC_ADX_error[batch_idx] < per_depth_auc_ae[d_preset]:
                            per_depth_auc_ae[d_preset] = AUC_ADX_error[batch_idx]  # 최소값 갱신
                    # Global AUC 계산
                    if sensor not in ae_list:
                        global_auc_non_ae_max = max(global_auc_non_ae_max, AUC_ADX_error[batch_idx])
                    else:
                        global_auc_ae_max = max(global_auc_ae_max, AUC_ADX_error[batch_idx])
                    global_auc_all_max = max(global_auc_all_max, AUC_ADX_error[batch_idx])
                

            if Det_Bbox == None:
                scores = [1 for x in range(len(estimated_Rs))]

            write_to_cvs.write_cvs(cvs_path, f"{obj_name}_{setting}", obj_id+1, scene_ids, sensor, d_preset, img_ids, estimated_Rs, estimated_Ts, ADX_error)
            

            print(f"-----{sensor}-----")
            AD5_passed = np.mean(AD5_passed)
            ADX_error_mean= np.mean(ADX_error)
            AUC_ADX_error = np.mean(AUC_ADX_error)

            if sensor not in ae_list:
                AD5_sensor_mean.append(AD5_passed)
                non_ae_add_means.append(ADX_error_mean)
                non_ae_r005.append(AD5_passed)
                per_depth_non_ae[d_preset]['add'].append(ADX_error_mean)
                per_depth_non_ae[d_preset]['r005'].append(AD5_passed)
            else:
                ae_add_means.append(ADX_error_mean)
                ae_r005.append(AD5_passed)
                per_depth_ae[d_preset]['add'].append(ADX_error_mean)
                per_depth_ae[d_preset]['r005'].append(AD5_passed)

            print('{}/{} 005'.format(main_metric_name,main_metric_name), AD5_passed)
            print('{}_error/{}'.format(main_metric_name,main_metric_name), ADX_error_mean)
            print('AUC_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX_error)
            AUC_ADX_error_posecnn = compute_auc_posecnn(ADX_error/1000.)
            print('AUC_posecnn_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX_error_posecnn)
            RE_error = np.mean(RE_error)
            TE_error = np.mean(TE_error)

            print('RE_error', RE_error)
            print('TE_error', TE_error)

            ####save results to file
            if has_gt:
                # path = os.path.join(eval_output_path, "ADD_result/")
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # path = path + "{}_{}".format(dataset_name, obj_name) + ".txt" 
                # #path = path + dataset_name + obj_name  + "ignorebit_" + str(configs['ignore_bit']) + ".txt"
                # #path = path + dataset_name + obj_name + "radix" + "_" + str(divide_number_each_itration)+"_"+str(number_of_itration) + ".txt"
                # print('save ADD results to', path)
                # print(path)
                # f = open(path, "w")
                # f.write('{}/{} '.format(main_metric_name,main_metric_name))
                # f.write(str(ADX_passed.item()))
                # f.write('\n')
                # f.write('AUC_{}/{} '.format(main_metric_name,main_metric_name))
                # f.write(str(AUC_ADX_error.item()))
                # f.write('\n')
                # f.write('AUC_posecnn_{}/{} '.format(main_metric_name,main_metric_name))
                # f.write(str(AUC_ADX_error_posecnn.item()))
                # f.write('\n')
                # f.write('AUC_{}/{} '.format(supp_metric_name,supp_metric_name))
                # f.write(str(AUC_ADY_error.item()))
                # f.write('\n')
                # f.write('AUC_posecnn_{}/{} '.format(main_metric_name,main_metric_name))
                # f.write(str(AUC_ADY_error_posecnn.item()))
                # f.write('\n')
                # f.close()

                f = open(path, "a")
                # 수정
                f.write(f"--- Depth {d_preset} | Sensor {sensor} ---\n")
                # 수정
                f.write(f"Preset_ADX_error_mean: {np.mean(preset_min_errors[d_preset]):.4f}\n")
                f.write(f'ADD_min_error: {str(np.mean(ADD_min_error))}\n')
                f.write('{}/{} 005 {}\n'.format(main_metric_name, main_metric_name, str(AD5_passed)))
                f.write('{}_error/{} {}\n'.format(main_metric_name, main_metric_name, str(ADX_error_mean)))
                f.write('AUC_{}/{} {}\n'.format(main_metric_name, main_metric_name, str(AUC_ADX_error)))
                f.write('AUC_posecnn_{}/{} {}\n'.format(main_metric_name, main_metric_name, str(AUC_ADX_error_posecnn)))
                f.write('RE_error {}\n'.format(str(RE_error)))
                f.write('TE_error {}\n'.format(str(TE_error)))
                f.close()
                ####




    non_ae_add_avg   = float(np.mean(non_ae_add_means))
    non_ae_r005_avg  = float(np.mean(non_ae_r005))

    ae_add_avg       = float(np.mean(ae_add_means))
    ae_r005_avg      = float(np.mean(ae_r005))

    global_oracle_add = float(np.mean(ADD_min_global))   

    success_ad5_img_ids = sorted(success_ad5_img_ids)
    print('----SUCCESS_IMG----')
    print('{}'.format(str(success_ad5_img_ids)))
    print('SUCCESS_AD5_IMG_NUM {}'.format(str(len(success_ad5_img_ids)/n_samples)))
    print('ADD_min_error {}'.format(str(np.mean(ADD_min_error))))
    print('RE_min_error {}'.format(str(np.mean(RE_min_error))))
    print('TE_min_error {}'.format(str(np.mean(TE_min_error))))
    print('ADD_sensor_mean 005 {}'.format(str(np.mean(AD5_sensor_mean))))
    print('{}'.format(str(best_rgb)))
    print('{}'.format(str(best_depth)))
    # print('USE_ICP {}'.format(str(configs['use_icp'])))
    f = open(path, "a")
    f.write('----SUCCESS_IMG----\n')
    f.write('{}\n'.format(str(success_ad5_img_ids)))
    f.write('SUCCESS_AD5_IMG_NUM {}\n'.format(str(len(success_ad5_img_ids)/n_samples)))
    f.write('ADD_min_error {}\n'.format(str(np.mean(ADD_min_error))))
    f.write('RE_min_error {}\n'.format(str(np.mean(RE_min_error))))
    f.write('TE_min_error {}\n'.format(str(np.mean(TE_min_error))))
    f.write('\nADD_sensor_mean 005 {}\n'.format(str(np.mean(AD5_sensor_mean))))
    f.write('{}\n'.format(str(best_rgb)))
    f.write('{}\n'.format(str(best_depth)))
    # f.write('USE_ICP {}\n'.format(str(configs['use_icp'])))
    # 수정
    f.write("\n--- Global Oracle (depth + sensor, AE 제외) ---\n")
    f.write(f"mean_ADD_global = {global_oracle_add:.4f}\n\n")

    f.write("--- Depth‑preset Oracle (AE 제외) ---\n")
    for dp in depth_preset_list:
        add_avg   = np.mean(per_depth_non_ae[dp]['add'])
        r005_avg  = np.mean(per_depth_non_ae[dp]['r005'])
        f.write(f"[Depth {dp}] non‑AE AVG  ADD:{add_avg:.4f}  r005:{r005_avg:.3f}\n")
        ae_add_avg   = np.mean(per_depth_ae[dp]['add'])
        ae_r005_avg  = np.mean(per_depth_ae[dp]['r005'])
        f.write(f"[Depth {dp}] AE AVG  ADD:{ae_add_avg:.4f}  r005:{ae_r005_avg:.3f}\n")
    for dp in depth_preset_list:
        f.write(f"Depth {dp} : ADD_mean={np.mean(preset_min_errors[dp]):.4f} "
                f"recall005={np.mean(depth_ad5_passed[dp]):.3f}\n")
        
        f.write(f"Depth {dp} : non ae AUC_mean={per_depth_auc_non_ae[dp]:.4f}\n")

        f.write(f"Depth {dp} : ae AUC_mean={per_depth_auc_ae[dp]:.4f}\n")

    f.write("\n--- Sensor‑level Averages ---\n")
    f.write(f"NON‑AE  | ADD {non_ae_add_avg:.4f}  r0.05 {non_ae_r005_avg:.3f}\n")
    f.write(f"AE‑only | ADD {ae_add_avg:.4f}      r0.05 {ae_r005_avg:.3f}\n")
    
    f.write(f"Global non-AE AUC (Max): {global_auc_non_ae_max:.4f}\n")
    f.write(f"Global AE AUC (Max): {global_auc_ae_max:.4f}\n")
    f.write(f"Global AUC (All Sensors, AE 포함): {global_auc_all_max:.4f}\n")
    
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str) # config file
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--region_bit', type=int, choices=range(16), default=10, help="the bit index")
    parser.add_argument('--threshold', type=float, default=50, help="inliers keep threshold, range from (0, 100), 50 equals np.median(), 70 means keep 70 precent of correspondences as inliers")
    parser.add_argument('--mean', action="store_true", help="use mean instead of np.median in outlier rejection threshold, in this case, threshold will not be used.")
    parser.add_argument('--uncertain_threshold', type=float, default=0.2, help="the probability in [0.5-uncetain_threshold, 0.5+uncetain_threshold] is viewed as uncertain")
    parser.add_argument('--eval_output_path', type=str)

    args = parser.parse_args()
    config_file = args.cfg
    checkpoint_file = args.ckpt_file
    eval_output_path = args.eval_output_path
    obj_name = args.obj_name
    configs = parse_cfg(config_file)
    configs['obj_name'] = obj_name

    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults

    configs['checkpoint_file'] = checkpoint_file
    configs['eval_output_path'] = eval_output_path

    configs['region_bit'] = args.region_bit
    configs['threshold'] = args.threshold
    configs['mean'] = args.mean

    #print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    main(configs)