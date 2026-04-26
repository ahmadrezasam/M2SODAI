#!/bin/bash

# Configuration
PROJECT_ROOT="/home/ahmadreza/Downloads/Research/M2SODAI"
YOLOV5_DIR="${PROJECT_ROOT}/yolov5_official"
DATA_DIR="${PROJECT_ROOT}/baseline_official"
WEIGHTS="${PROJECT_ROOT}/yolov5s.pt"
HYP="${DATA_DIR}/hyp_paper.yaml"
OUTPUT_ROOT="${DATA_DIR}/runs/seed42"

# Ensure output directory exists
mkdir -p "${OUTPUT_ROOT}"

# Function to train a fold
train_fold() {
    local fold_num=$1
    local fold_name="fold_${fold_num}"
    local data_file="${DATA_DIR}/${fold_name}.yaml"
    
    echo "=========================================================="
    echo "STARTING TRAINING FOR FOLD ${fold_num} (SEED 42)"
    echo "=========================================================="
    
    python3 "${YOLOV5_DIR}/train.py" \
        --weights "${WEIGHTS}" \
        --cfg "${YOLOV5_DIR}/models/yolov5s.yaml" \
        --data "${data_file}" \
        --hyp "${HYP}" \
        --img 640 \
        --batch-size 8 \
        --workers 0 \
        --epochs 300 \
        --patience 100 \
        --single-cls \
        --seed 42 \
        --project "${OUTPUT_ROOT}" \
        --name "${fold_name}" \
        --exist-ok
        
    echo "FINISHED TRAINING FOR FOLD ${fold_num}"
}

# Run all 3 folds sequentially
train_fold 1
train_fold 2
train_fold 3

echo "=========================================================="
    echo "ALL FOLDS COMPLETED"
echo "=========================================================="
