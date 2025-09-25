#!/bin/bash

# 고정 파라미터
obj_ids=(0 1 2 3 4)  # spray: 0, pringles: 1, tincase: 2, sandwich: 3, mouse: 4
depth_levels=(0 1 2 3)   # 0, 1, 2, 3
base_path=/ssd/sjkim/SAM-6D/Data/SS6D

# Run pose estimation model
cd /ssd/sjkim/SAM-6D/Pose_Estimation_Model

for obj_id in "${obj_ids[@]}"; do
    echo "STARTED Running pose estimation for object ID: $obj_id"
    
    # cad model number 계산은 루프 안에서 수행해야 함
    cad_model_number=$(printf "%06d" $((obj_id + 1)))
    TEMPLATE_PATH=/ssd/sjkim/SAM-6D/Data/SS6D-Templates/obj_${cad_model_number}

    for depth_level in "${depth_levels[@]}"; do
        echo "STARTED Running for obj: ${obj_id}, depth: ${depth_level}"
        
        # Run inference
        python run_inference_ss6d_whole.py \
            --base_path $base_path \
            --template_path $TEMPLATE_PATH \
            --obj_id $obj_id \
            --depth_level $depth_level 

    done
done
