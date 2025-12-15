#!/bin/bash

# 고정 파라미터
obj_id=(0 1 2 3 4)      # spray: 0, pringles: 1, tincase: 2, sandwich: 3, mouse: 4
depth_levels=(0 1 2 3)   # 0, 1, 2, 3
base_path=/ssd/sjkim/SAM-6D/Data/SS6D
SEGMENTOR_MODEL=sam

cad_model_number=$(printf "%06d" $((obj_id + 1)))  # spray: 1, pringles: 2, tincase: 3, sandwich: 4, mouse: 5
TEMPLATE_PATH=/ssd/sjkim/SAM-6D/Data/SS6D-Templates/obj_${cad_model_number}

# CAD_PATH=${base_path}/models/obj_${cad_model_number}.ply
# cd ~/SAM-6D/Render
# blenderproc run --custom-blender-path /ssd/sjkim/SAM-6D/blender-3.3.1-linux-x64 \
#     render_ss6d_templates.py --template_path $TEMPLATE_PATH --cad_path $CAD_PATH

# Run instance segmentation model
cd /ssd/sjkim/SAM-6D/Instance_Segmentation_Model
for depth_level in "${depth_levels[@]}"; do
    python run_inference_ss6d_whole.py \
        --segmentor_model $SEGMENTOR_MODEL \
        --base_path $base_path \
        --template_path $TEMPLATE_PATH \
        --obj_id $obj_id \
        --depth_level $depth_level 
done