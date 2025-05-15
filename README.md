# SenseShift6D
## 📦 Dataset Download & Extraction
### Step 0: Download Dataset Zip Files
Our dataset is hosted on [Hugging Face](https://huggingface.co/datasets/Yegyu/SenseShift6D) and can be downloaded using the link below:

[📥 Download SenseShift6D Dataset](https://huggingface.co/datasets/Yegyu/SenseShift6D)

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

___

This dataset is released under the **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)** license.  
You are free to use, share, and adapt the dataset, provided that appropriate credit is given.
