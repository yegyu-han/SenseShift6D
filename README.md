# SenseShift6D
This repository provides the **SenseShift6D** dataset for evaluating 6D object pose estimation under varying sensor and lighting conditions.

---

## 📦 Dataset Download & Extraction
### Step 0: Download Dataset Zip Files
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
```
SenseShift6D/
├── train/
│   └── Bx/                               # Brightness level (e.g., B5, B25, B50, B75, B100)
│       └── 000000/                       # Scene number
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

## ⚠️ Requirement

### 🧪 Physical Based Rendering for Synthetic Training
We provide a synthetic training set rendered with physically-based rendering (PBR):

```
SenseShift6D/
└── train_pbr/
        └── obj1/
        └── ... /
```

This was generated using [BlenderProc](https://github.com/DLR-RM/BlenderProc) following the official BOP PBR rendering instructions.

### 🧪 Ground Truth Mesh for ZebraPose & HiPose
ZebraPose and HiPose require GT-colored meshes and pre-generated GT files for training and evaluation on custom datasets like SenseShift6D.

To generate these files, you must first convert the CAD models into colored meshes and create ground-truth label files (e.g., train_GT, test_GT).

You can follow the instructions from the [ZebraPose repository](https://github.com/suyz526/ZebraPose) under Binary_Code_GT_Generator/ to generate:

- models_GT_color/: colored mesh models (e.g., .ply with per-vertex GT color)

- train_GT/, test_GT/: ground truth binary code labels

Please ensure the following directory structure exists:

```
SenseShift6D/
├── models_GT_color/
├── train_GT/
├── train_pbr_GT/
└── test_GT/
```

___

## 🔧 Baseline Models
SenseShift6D is designed to be easily integrated into various existing 6D object pose estimation frameworks. In our experiments, we evaluated the dataset using three popular baseline models:

### 📌 GDRNPP
Original Repository: [GDRNPP_BOP2022](https://github.com/shanice-l/gdrnpp_bop2022)

Modifications:

Added core/configs/sense_shift6d/*.yaml

Added lib/datasets/sense_shift6d_dataset.py

Set DATASETS.DATA_ROOT to the SenseShift6D/ directory in the config files

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

Added configs/ss6d.yaml and tools/train_ss6d.py

Added dataset/sense_shift6d_dataset.py

Set DATA.ROOT to the SenseShift6D/ directory in the config file

Each model can be run with minimal changes to the original codebase. You may either manually copy the modified files or use our provided script to automatically patch the respective repositories.

---

## 📄 License
This dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
You are free to share and adapt the dataset with proper attribution.
