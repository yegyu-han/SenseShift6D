# SenseShift6D
This repository provides the **SenseShift6D** dataset for evaluating 6D object pose estimation under varying sensor and lighting conditions.

---

<!---
## 📦 Dataset Download & Extraction
### Step 0: Download Dataset Zip Files --
Download the following files from our [Hugging Face repository 🤗](https://huggingface.co/datasets/Yegyu/SenseShift6D):
- `ss6d_base.zip`
- `scene_000000.zip`
- `scene_000001.zip`
- `scene_000002.zip`
  
To use the SenseShift6D dataset, follow the steps below:

### Step 1: Unzip Base Archive
Unzip the base archive `ss6d_base.zip`
```
unzip ss6d_base.zip
```

### Step 2: Unzip Each Scene
Use the provided `unzip_tool.py` script to extract each scene (`scene_000000.zip`, `scene_000001.zip`, `scene_000002.zip`) archive into the dataset directory.
```
## Example Usage
python unzip_tool.py --zip_path ./scene_000000.zip --output_dir ./SenseShift6D
```
After extraction, you will have a structured dataset under SenseShift6D/train/ and/or SenseShift6D/test/.

## 📁 Final Directory Structure
After extraction, your SenseShift6D/ directory should be structured as follows:

## 📦 Dataset Download
Download the SenseShfit6D from our [Hugging Face repository 🤗](https://huggingface.co/datasets/Yegyu/SenseShift6D)

## 📁 SenseShift6D Directory Structure
```
SenseShift6D/
├── train/
│   └── Bx/                               # Brightness level (e.g., B5, B25, B50, B75, B100)
│       └── 000000/                      
│           ├── rgb/
│           │   └── EXGX/                 # Exposure-Gain setting (e.g., AE, E4G32)
│           │       └── *.png             
│           ├── depth/
│           │   └── capture_mode/         # Depth capture mode (0: default, 1: high accuracy, 2: high density, 3: medium density)
│           │       └── *.png             
│           ├── mask/
│           │   └── *.png                 
│           ├── mask_visib/
│           │   └── *.png                 
│           ├── scene_camera.json         
│           ├── scene_gt.json             
│           └── scene_gt_info.json        
│
├── test/
│   └── Bx/
│       └── ... (same structure as train/)
│
├── models/
│   ├── obj_*.ply                     
│   └── models_info.json          
│
├── models_eval/
│   └── ... (same structure as models/)
│
└── camera.json                                             

```
___

-->

## ⚠️ Requirements for Baseline Models

### 🧪 Physical Based Rendering for Synthetic Training
We generated our synthetic PBR images using the [BlenderProc4BOP](https://github.com/DLR-RM/BlenderProc/blob/main/README_BlenderProc4BOP.md). The resulting folder layout is:

```
SenseShift6D/
└── train_pbr/
    └── obj1/
    └── obj2/
    └── obj3/
```
For a minimal example of how to invoke BOPToolkit on your own CAD models, see our helper script in tools/main_custom_upright.py and tools/main_custom_surface.py

### 🧪 Detection File for GDRNPP Testing
To generate detection file for gdrnpp, run [this code](https://github.com/yegyu-han/SenseShift6D/blob/main/tools/scene_info_to_det_file.py)

### 🧪 Ground Truth Mesh for ZebraPose & HiPose
ZebraPose and HiPose require GT-colored meshes and pre-generated GT files for training and evaluation on custom datasets like SenseShift6D.

To generate these files, you must first convert the CAD models into colored meshes and create ground-truth label files (e.g., train_GT, test_GT).

You can follow the instructions from the [ZebraPose repository](https://github.com/suyz526/ZebraPose) under Binary_Code_GT_Generator/ to generate:

- models_GT_color/: colored mesh models (e.g., .ply with per-vertex GT color)

- train_GT/, test_GT/: ground truth binary code labels

Please ensure the following directory structure exists:

```
SenseShift6D/
├── models_GT_color/     # Colored meshes (.ply) with GT labels
├── train_GT/            # GT labels for real training images
├── train_pbr_GT/        # GT labels for synthetic training images
└── test_GT/             # GT labels for test images
```

___

## 🔧 Baseline Models (Instance-level)
SenseShift6D is designed to be easily integrated into various existing 6D object pose estimation frameworks. In our experiments, we trained three popular baseline models with the dataset:

### 📌 GDRNPP
Original Repository: [GDRNPP_BOP2022](https://github.com/shanice-l/gdrnpp_bop2022)

Modifications:

- Added:

  - configs/gdrn/ss6dSO/*.py
    
  - core/gdrn_modeling/datasets/ss6d_*.py
  
  - ref/ss6d.py  

- Modified:
  
  - core/gdrn_modeling/engine/gdrn_custom_evaluator.py  
 
  - core/gdrn_modeling/engine/engine.py  

### Training:

```
./core/gdrn_modeling/train_gdrn.sh config/gdrn/ss6dSO/01_spray.py <gpu_ids> (other args)
```

### Testing:

```
./core/gdrn_modeling/test_gdrn.sh config/gdrn/ss6dSO/01_spray.py <gpu_ids> output/gdrn/SS6D/exp1/spray/model_final.pth
```

### 📌 ZebraPose
Original Repository: [ZebraPose](https://github.com/suyz526/ZebraPose)

Modifications:

- Added:

   - ZebraPose/zebrapose/configs/config_SS6D/

   - ZebraPose/zebrapose/outputs/checkpoints/

   - ZebraPose/zebrapose/tools_for_BOP/ss6d_io.py

   - ZebraPose/zebrapose/ss6d_dataset_pytorch.py

   - ZebraPose/zebrapose/SS6D_Augmentation.py

- Modified:

   - ZebraPose/zebrapose/tools_for_BOP/common_dataset_info.py

   - ZebraPose/zebrapose/config_parser.py

### Training:

```
python train_ss6d.py --cfg config/config_SS6D/exp_SS6D_train_general.txt --obj_name spray
```

### Testing:

```
python test_ss6d_f.py \
  --cfg config/config_SS6D/exp_SS6D_test_AE_B5.txt \
  --obj_name spray \
  --ckpt_file outputs/checkpoints/exp_SS6D_train_general/spray \
  --ignore_bit 0 \
  --eval_output_path outputs/report
```

### 📌 HiPose
Original Repository: [HiPose](https://github.com/lyltc1/HiPose)

Modifications:

- Added:

   - HiPose/hipose/config/
     
   - HiPose/hipose/tools_for_BOP/ss6d_io.py
 
   - HiPose/hipose/tools_for_BOP/ss6d_io_depth_oracle.py
 
   - HiPose/hipose/tools_for_BOP/pbr_io.py
 
   - HiPose/hipose/train_ss6d.py
 
   - HiPose/hipose/test_with_region_v3.py
 
   - HiPose/hipose/test_with_region_v3_depth_oracle.py
 
- Modified:

   - HiPose/hipose/bop_dataset_3d_convnext_backbone.py
 
   - HiPose/hipose/config_parser.py
 
   - HiPose/hipose/GDR_Net_Augmentation.py
 
   - HiPose/hipose/tools_for_BOP/common_dataset_info.py


### Training:

```
python train_ss6d.py --cfg config/train_senseshift6d_config_general.txt --obj_name spray
```

### Testing:

```
python test.py \
  --cfg config/test_senseshift6d_config_B5.txt \
  --obj_name spray \
  --ckpt_file outputs/checkpoints/~ \
  --eval_output outputs \
  --new_solver_version True \
  --region_bit 10
```

Set DATA.ROOT to the SenseShift6D/ directory in the config file


Each model can be run with minimal changes to the original codebase. You may either manually copy the modified files or use our provided script to automatically patch the respective repositories.


---
## 🔧 Baseline Models (Unseen)

### 📌 SAM-6D
Original Repository: [SAM-6D](https://github.com/JiehongLin/SAM-6D)

Modifications:

- Added:

   - sam6d/ISM/run_inference_ss6d_whole.py

   - sam6d/PEM/run_inference_ss6d_whole.py
 
   - sam6d/PEM/run_inference_ss6d_whole_gt.py
   
   - sam6d/scripts/evaluation/calculate_ADD_ss6d.py

   - sam6d/scripts/evaluation/calculate_ADD_ss6d_gt.py

   - sam6d/scripts/evaluation/eval.sh
   
   - sam6d/scripts/evaluation/merge.py

   - sam6d/scripts/run/demo_ss6d_whole_ism.sh

   - sam6d/scripts/run/demo_ss6d_whole_pem.sh

   - sam6d/scripts/run/demo_ss6d_whole_pem_gt.sh


### Instance Segmentation Model:

First place `sam6d/ISM/run_inference_ss6d_whole.py` under original [SAM-6D repository](https://github.com/JiehongLin/SAM-6D) -> `SAM-6D/Instance_Segmentation_Model`.

You can render CAD templates by uncommentting the lines before "Run instance segementation model" in the file.

```
./sam6d/scripts/run/demo_ss6d_whole_ism.sh
```

### Pose Estimation Model:

First place `am6d/PEM/run_inference_ss6d_whole.py, am6d/PEM/run_inference_ss6d_whole_gt.py` under original [SAM-6D repository](https://github.com/JiehongLin/SAM-6D) -> `SAM-6D/Pose_Estimation_Model`.

You can choose `demo_ss6d_whole_pem.sh` to use ism segmentation mask or `demo_ss6d_whole_pem_gt.sh` to use GT segmentation mask.

```
./sam6d/scripts/run/demo_ss6d_whole_pem.sh
```

### Evaluation:

Calculates ADD and AUC values for pem results.

```
./sam6d/scripts/evaluation/eval.sh
```


### 📌 FoundationPose
Original Repository: [FoundationPose](https://github.com/NVlabs/FoundationPose)

Modifications:

- Added:

   - FoundationPose/run_ss6d.py
 
   - FoundationPose/run_ss6d_depth_oracle.py
 
   - FoundationPose/datareader_depth_oracle.py
 
- Modified:

   - FoundationPose/datareader.py


### Testing:

```
python run_ss6d.py \
  --brightness B75 \
  --general False \
  --obj_name spray \
  --eval_output_path outputs \
  --debug 0
```



## ⚒️ Tools
Some useful codes.

+ [calc_ann_errs.py](/tools/calc_ann_errs.py): You can calculate GT annotation accuracy. Place it under *gdrnpp_bop2022/core/gdrn_modeling/tools/ss6d* in [GDRNPP_BOP2022](https://github.com/shanice-l/gdrnpp_bop2022).

+ [main_custom_surface.py](/tools/main_custom_surface.py): PBR generator placing objects with random poses. Place it under *BlenderProc/examples/datasets/bop_challenge* in [BlenderProc](https://github.com/DLR-RM/BlenderProc)

+ [main_custom_upright.py](/tools/main_custom_upright.py): PBR generator placing objects with upright poses.

+ [custom_v1_vis_poses_full_gdrn.py](/tools/custom_v1_vis_poses_full_gdrn.py): Visualization of GDRNPP estimated pose (green) & Gt pose (red). Place it under *gdrnpp_bop2022/core/gdrn_modeling/tools/ss6d* in [GDRNPP_BOP2022](https://github.com/shanice-l/gdrnpp_bop2022).

+ [custom_v1_vis_poses_full_hipose_depth.py](/tools/custom_v1_vis_poses_full_hipose_depth.py): Visualization of HiPose estimated pose (green) & Gt pose (red). Place it under *gdrnpp_bop2022/core/gdrn_modeling/tools/ss6d* in [GDRNPP_BOP2022](https://github.com/shanice-l/gdrnpp_bop2022).

---

## 📄 License
This repository is primarily licensed under the **MIT License**.

However, it includes two files adapted from [BlenderProc](https://github.com/DLR-RM/BlenderProc), which is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.  
As such, **only the following files are subject to GPL-3.0**:

- [`tools/main_custom_upright.py`](https://github.com/yegyu-han/SenseShift6D/blob/main/tools/main_custom_upright.py)
- [`tools/main_custom_surface.py`](https://github.com/yegyu-han/SenseShift6D/blob/main/tools/main_custom_surface.py)

### Components adapted from third-party sources:
  - **BlenderProc** (GPL-3.0): rendering tools under `tools/`
  - **GDRNPP** (Apache-2.0): modules under `gdrnpp/`
  - **ZebraPose** (MIT): modules under `zebrapose/`
  - **HiPose** (MIT): modules under `hipose/`

All external components are used in accordance with their respective licenses. The full MIT license text can be found in the `LICENSE` file.
