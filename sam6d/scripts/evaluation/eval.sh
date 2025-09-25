#!/bin/bash
# Usage: ./eval.sh [gt|top1|default]

if [ -z "$1" ]; then
  echo "[ERROR] Enter a option: $0 [gt|top1|default]"
  exit 1
fi

# 고정 파라미터
obj_ids=(0 1 2 3 4)  # spray: 0, pringles: 1, tincase: 2, sandwich: 3, mouse: 4
base_path=/ssd/sjkim/SAM-6D/Data/SS6D
brightnesses=(B5 B25 B50 B75 B100)

echo " ... MERGING PEM RESULTS ... "

cd scripts/evaluation
for brightness in "${brightnesses[@]}"; do

    for obj_id in "${obj_ids[@]}"; do

        if [ "$1" = "gt" ]; then
            python ./merge.py --obj_id $obj_id --b_level $brightness --gt_detection True
        elif [ "$1" = "top1" ]; then
            python ./merge.py --obj_id $obj_id --b_level $brightness --top1 True
        elif [ "$1" = "default" ]; then        # default: runs when option is 'default'
            python ./merge.py --obj_id $obj_id --b_level $brightness 
        else 
            echo "[ERROR] Not a valid option: $1"  
            exit 1
        fi
    done

    echo " ... CALCULATING AUC SCORE ... "
    for obj_id in "${obj_ids[@]}"; do
        if [ "$1" = "gt" ]; then
            python ./calculate_ADD_ss6d_gt.py --object_id $obj_id --b_level $brightness 
        elif [ "$1" = "top1" ]; then
            python ./calculate_ADD_ss6d_top1.py --object_id $obj_id --b_level $brightness 
        elif [ "$1" = "default" ]; then
            python ./calculate_ADD_ss6d.py --object_id $obj_id --b_level $brightness 
        fi
    done
    
done
