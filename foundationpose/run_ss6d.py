# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import sys, os

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, "bop_toolkit"))
from bop_toolkit.bop_toolkit_lib import inout, pose_error
from write_csv import write_csv


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

def Calculate_ADD_Error_BOP(R_GT,t_GT, R_predict, t_predict, vertices):
    t_GT = t_GT.reshape((3,1))
    t_predict = np.array(t_predict).reshape((3,1))

    return pose_error.add(R_predict, t_predict, R_GT, t_GT, vertices)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_obj', type=str, default=None)
  parser.add_argument('--mesh_ply', type=str, default=None)
  parser.add_argument('--dataset_dir', type=str, default='/hdd/yghan/SenseShift6D/test') 
  parser.add_argument('--brightness', type=str, default='B50')
  parser.add_argument('--depth_shuffle', action='store_true')
  parser.add_argument('--sensor', type=str, default=None)
  parser.add_argument('--general', type=str, default='False', help='Use general sensor loop or not (True/False)')
  parser.add_argument('--obj_id', type=int, default=0)
  parser.add_argument('--obj_name', type=str, required=True)
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--eval_output_path', type=str)
  parser.add_argument('--bop_path', type=str, default='/hdd/yghan')
  parser.add_argument('--dataset_name', type=str, default='SenseShift6D')
  parser.add_argument('--test_folder', type=str, default='test')
  parser.add_argument('--det_json_path', type=str, default=None, help="Detection 결과 JSON 파일 경로")
  
  args = parser.parse_args()

  OBJ_NAME_TO_ID = {
    "spray": 0,
    "pringles": 1,
    "tincase": 2,
    "sandwich": 3,
    "mouse": 4,
    "duck": 5
  } 
  args.obj_id = OBJ_NAME_TO_ID[args.obj_name]

  sensor_list = [
                'AE',
                'E9G16', 'E9G48', 'E9G80', 'E9G112',
                'E39G16', 'E39G48', 'E39G80', 'E39G112',
                'E156G16', 'E156G48', 'E156G80', 'E156G112',
                'E625G16', 'E625G48', 'E625G80', 'E625G112',
                'E2500G16', 'E2500G48', 'E2500G80', 'E2500G112']
    
  ae_list = ['AE', 'AEG16', 'AEG48', 'AEG80', 'AEG112']


  if args.mesh_ply is None:
    mesh_obj_id = args.obj_id + 1  # 1부터 시작
    mesh_ply_path = os.path.join('/hdd/yghan/SenseShift6D/models', f'obj_{mesh_obj_id:06d}.ply')
    args.mesh_ply = mesh_ply_path
    print(f"[mesh_ply 자동 생성] obj_id + 1 → {mesh_obj_id:06d}")
  else:
    print(f"[mesh_ply 직접 지정]: {args.mesh_ply}")

  if args.mesh_obj is None:
    mesh_obj_id = args.obj_id + 1  # 1부터 시작
    mesh_obj_path = os.path.join('/hdd/yghan/SenseShift6D/models', f'obj_{mesh_obj_id:06d}.obj')
    args.mesh_obj = mesh_obj_path
    print(f"[mesh_obj 자동 생성] obj_id + 1 → {mesh_obj_id:06d}")
  else:
    print(f"[mesh_obj 직접 지정]: {args.mesh_obj}")

  print(f"최종 사용될 mesh_obj 경로: {args.mesh_obj}")
  print(f"최종 사용될 mesh_ply 경로: {args.mesh_ply}")

  vertices = inout.load_ply(args.mesh_ply)["pts"] / 1000

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_obj)
  mesh.vertices *= 0.001 # 반드시 추가!! mm -> m 

  debug = args.debug
  debug_dir = args.debug_dir
  # os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
  base_debug_dir = os.path.join(args.debug_dir, args.brightness, args.obj_name)
  os.makedirs(base_debug_dir, exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  refiner.cfg['use_normal'] = False # 법선벡터 사용 X
  refiner.cfg['use_light'] = False # 법선벡터 사용 X
  glctx = dr.RasterizeCudaContext()

  eval_output_path = args.eval_output_path
  brightness = args.brightness
  obj_id = args.obj_id
  obj_name = args.obj_name
  bop_path = args.bop_path
  dataset_name = args.dataset_name
  test_folder = args.test_folder
  args.general = args.general.lower() in ['true', 'True']

  model_dir = bop_path+"/"+dataset_name+"/models"
  model_info = inout.load_json(os.path.join(model_dir,"models_info.json"))
  obj_diameter = model_info[str(obj_id+1)]['diameter'] / 1000
  print("obj_diameter: ", obj_diameter)

  if not os.path.exists(eval_output_path):
      os.makedirs(eval_output_path)
  path = os.path.join(eval_output_path, "{}_{}_{}.txt".format(dataset_name, obj_name, brightness))
  if os.path.exists(path):
      os.remove(path)

  success_ad2_img_ids = set()
  success_ad5_img_ids = set()
  AD2_sensor_mean = []
  AD5_sensor_mean = []
  sample_path = os.path.join(bop_path, dataset_name, test_folder, brightness, str(obj_id).zfill(6), 'rgb/AE')
  n_samples = len(glob.glob(os.path.join(sample_path, '*.png')))
  ADD_min_error = np.ones(n_samples) * 10000
  RE_min_error = np.ones(n_samples) * 10000
  TE_min_error = np.ones(n_samples) * 10000


  for sensor in sensor_list:
    debug_dir = os.path.join(base_debug_dir, sensor)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'track_vis'), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'ob_in_cam'), exist_ok=True)

    est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=debug_dir,
    debug=debug,
    glctx=glctx
    )
    logging.info("estimator initialization done")

    print("======== 세팅 확인 ========")
    print(f"obj_name: {args.obj_name}")
    print(f"Brightness: {args.brightness}")
    print(f"Sensor: {sensor}")
    print(f"depth_shuffle: {args.depth_shuffle}")
    print("=========================")

    cvs_path = os.path.join(eval_output_path, brightness, obj_name)
    os.makedirs(cvs_path, exist_ok=True)
    cvs_path = os.path.join(cvs_path, "{}_{}_{}_{}".format(dataset_name, obj_name, brightness, sensor))

    # if sensor in ['AEG16', 'AEG48', 'AEG80', 'AEG112']:
    #         continue

    # 추가
    # json_base_dir = "/ssd/dywoo/FoundationPose/sam_mask"
    # json_filename = f"merged_ism_topscore_AE_{obj_name}_{brightness}_d0.json"
    # det_json_path = os.path.join(json_base_dir, json_filename)
    det_json_path = None

    
    print("SS6DReader 초기화 중...")  
    reader = SS6DReader(
      root_dir=args.dataset_dir,
      brightness=args.brightness,
      general=args.general,
      sensor=sensor,
      depth_shuffle=args.depth_shuffle,
      obj_id=args.obj_id,
      use_detection_mask = False,
      det_json_path = det_json_path
    )
    print("SS6DReader 초기화 완료.")

    print(f"총 프레임 수: {len(reader)}\n")

    AD2_passed=np.zeros(len(reader))
    AD5_passed=np.zeros(len(reader))
    RE_error=np.zeros(len(reader))
    TE_error=np.zeros(len(reader))
    ADX_error=np.zeros(len(reader))
    AUC_ADX_error=np.zeros(len(reader))
    pose_list = [None] * len(reader)  
    pose_score_list = []

    for i in range(len(reader)):
      print(f'프레임 i: {i}')
      color = reader.get_color(i)
      depth = reader.get_depth(i)
      K = reader.get_K(i)
      mask = reader.get_mask(i).astype(bool)
      if np.sum(mask) == 0:
        print(f"[경고] 마스크가 비어 있음: frame i={i}, img_id={reader.ids[i]}")

      r_GT, t_GT = reader.get_GT(i)

      print(f"[{obj_name}] 이미지 ID: {int(reader.ids[i]):06d}")
      print(f"RGB: {reader.rgb_files[i]}")
      print(f"Depth: {reader.depth_files[i]}")
      print(f"Mask: {reader.mask_files[i]}")
      print(f"Intrinsic K: \n{K}")

      # gt_mask = imageio.imread(reader.mask_files[i]).astype(bool)

      # 기존에는 0번 이미지에서 Pose estimation 후 tracking 하는 과정이었는데,
      # SS6D는 두 이미지 간 시간 차가 큰 편이라 tracking 없이 pose estimation 수행하도록 함 
      pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      pose_score = est.scores[0].item()
      R_predict, t_predict = pose[:3,:3], pose[:3,3]
      adx_error = Calculate_ADD_Error_BOP(r_GT, t_GT, R_predict, t_predict, vertices)
      pose_list[i] = pose
      print(f"[추적 완료] Pose:\n{pose}")

      pose_score_list.append(pose_score)

      re_error = re(r_GT, R_predict)
      te_error = te(t_GT, t_predict)
      if np.isnan(adx_error): adx_error = 10000
      if np.isnan(re_error): re_error = 10000
      if np.isnan(te_error): te_error = 10000

      if adx_error < obj_diameter*0.05:
          AD5_passed[i] = 1
          if sensor not in ae_list:
              success_ad5_img_ids.add(f"{int(reader.ids[i]):06d}")

      ADX_error[i] = adx_error
      RE_error[i] = re_error
      TE_error[i] = te_error

      if adx_error < ADD_min_error[i] and sensor not in ae_list:
          ADD_min_error[i] = adx_error
          RE_min_error[i] = re_error
          TE_min_error[i] = te_error

      print(f"[결과] 프레임{i}, adx_error {adx_error}")

      th = np.linspace(0, 0.10, num=100)
      sum_correct = 0
      for t in th:
          if adx_error < obj_diameter*t:
              sum_correct = sum_correct + 1
      AUC_ADX_error[i] = sum_correct/100
      
      ob_in_cam_path = os.path.join(debug_dir, 'ob_in_cam')
      os.makedirs(ob_in_cam_path, exist_ok=True)
      np.savetxt(f'{ob_in_cam_path}/{int(reader.ids[i]):06d}.txt', pose.reshape(4, 4))

      if debug >= 1:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)

      if debug >= 2:
        vis_save_dir = os.path.join(debug_dir, 'track_vis')
        os.makedirs(vis_save_dir, exist_ok=True)
        vis_path = os.path.join(vis_save_dir, f'{int(reader.ids[i]):06d}.png')
        imageio.imwrite(vis_path, vis)

    write_csv(cvs_path,
              obj_id=obj_id,
              scene_id_=[0]*len(reader), 
              sensor=sensor,
              depth=0,  # depth_shuffle 안 쓰면 그냥 0으로
              img_id_=[int(i) for i in reader.ids],
              pose_=pose_list,  # 각 프레임에서 추정한 4x4 pose
              scores=ADX_error.tolist(),  # 각 프레임의 adx
              pose_scores = pose_score_list
              )


    print(f"-----{sensor}-----")
    AD5_passed_oracle = np.mean(AD5_passed)
    if sensor not in ae_list:
        AD5_sensor_mean.append(AD5_passed_oracle)
    ADX_error_mean= np.mean(ADX_error)
    AUC_ADX_error_mean = np.mean(AUC_ADX_error)
    print('{}/{} 005'.format('ADD','ADD'), AD5_passed_oracle)
    print('{}_error/{}'.format('ADD','ADD'), ADX_error_mean)
    print('AUC_{}/{}'.format('ADD','ADD'), AUC_ADX_error_mean)
    RE_error_mean = np.mean(RE_error) 
    TE_error_mean = np.mean(TE_error)
    print('RE_error', RE_error_mean)
    print('TE_error', TE_error_mean)

    f = open(path, "a")
    f.write('-----{}-----\n'.format(sensor))
    f.write('{}/{} 005 {}\n'.format('ADD', 'ADD', str(AD5_passed_oracle)))
    f.write('{}_error/{} {}\n'.format('ADD', 'ADD', str(ADX_error_mean)))
    f.write('AUC_{}/{} {}\n'.format('ADD', 'ADD', str(AUC_ADX_error_mean)))
    f.write('RE_error {}\n'.format(str(RE_error_mean)))
    f.write('TE_error {}\n'.format(str(TE_error_mean)))
    f.close()
    ####

  success_ad5_img_ids = sorted(success_ad5_img_ids)
  print('----SUCCESS_IMG----')
  print('{}'.format(str(success_ad5_img_ids)))
  print('SUCCESS_AD5_IMG_NUM {}'.format(str(len(success_ad5_img_ids)/n_samples)))
  print('ADD_min_error {}'.format(str(np.mean(ADD_min_error))))
  print('RE_min_error {}'.format(str(np.mean(RE_min_error))))
  print('TE_min_error {}'.format(str(np.mean(TE_min_error))))
  print('ADD_sensor_mean 005 {}'.format(str(np.mean(AD5_sensor_mean))))

  f = open(path, "a")
  f.write('----SUCCESS_IMG----\n')
  f.write('{}\n'.format(str(success_ad5_img_ids)))
  f.write('SUCCESS_AD5_IMG_NUM {}\n'.format(str(len(success_ad5_img_ids)/n_samples)))
  f.write('ADD_min_error {}\n'.format(str(np.mean(ADD_min_error))))
  f.write('RE_min_error {}\n'.format(str(np.mean(RE_min_error))))
  f.write('TE_min_error {}\n'.format(str(np.mean(TE_min_error))))
  f.write('ADD_sensor_mean 005 {}'.format(str(np.mean(AD5_sensor_mean))))
  f.close()
