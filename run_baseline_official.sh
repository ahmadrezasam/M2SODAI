#!/usr/bin/env bash
# ==============================================================================
# Train YOLOv5s Lower Bound Baseline – 3-Fold Cross-Validation
#
# Paper: Sakacı et al. (2024) – Optical (RGB) trained → HSI tested
# Memory-safe settings: batch-size 8, workers 0 (for 15GB RAM systems)
# Uses OFFICIAL dataset HSI-RGB and HSI-PCA test images
# ==============================================================================

set -e

PROJECT_ROOT="/home/ahmadreza/Downloads/Research/M2SODAI"
YOLOV5_DIR="${PROJECT_ROOT}/yolov5_official"
DATA_DIR="${PROJECT_ROOT}/baseline_official"
WEIGHTS="${PROJECT_ROOT}/yolov5s.pt"
HYP="${DATA_DIR}/hyp_paper.yaml"

echo "============================================"
echo "  Lower Bound Baseline Training"
echo "  Using Official YOLOv5 Repository"
echo "============================================"

# Train each fold
for FOLD in 1 2 3; do
    FOLD_DIR="${DATA_DIR}/runs/fold_${FOLD}"
    BEST_PT="${FOLD_DIR}/weights/best.pt"

    # Skip if already trained
    if [ -f "${BEST_PT}" ]; then
        echo ""
        echo "  Fold ${FOLD} already trained (${BEST_PT} exists). Skipping."
    else
        echo ""
        echo "--------------------------------------------"
        echo "  Training Fold ${FOLD}/3"
        echo "--------------------------------------------"

        python3 "${YOLOV5_DIR}/train.py" \
            --weights "${WEIGHTS}" \
            --cfg "${YOLOV5_DIR}/models/yolov5s.yaml" \
            --data "${DATA_DIR}/fold_${FOLD}.yaml" \
            --hyp "${HYP}" \
            --img 640 \
            --batch-size 8 \
            --workers 0 \
            --epochs 300 \
            --patience 100 \
            --single-cls \
            --project "${DATA_DIR}/runs" \
            --name "fold_${FOLD}" \
            --exist-ok

        echo "  Fold ${FOLD} training complete."
    fi
done

echo ""
echo "============================================"
echo "  All folds trained. Starting evaluation..."
echo "============================================"

# Evaluate each fold on OFFICIAL HSI-RGB and HSI-PCA test sets
for FOLD in 1 2 3; do
    BEST_WEIGHTS="${DATA_DIR}/runs/fold_${FOLD}/weights/best.pt"

    echo ""
    echo "--------------------------------------------"
    echo "  Evaluating Fold ${FOLD} on Official HSI-RGB test"
    echo "--------------------------------------------"

    python3 "${YOLOV5_DIR}/val.py" \
        --weights "${BEST_WEIGHTS}" \
        --data "${DATA_DIR}/test_official_hsi_rgb.yaml" \
        --img 640 \
        --batch-size 32 \
        --single-cls \
        --project "${DATA_DIR}/eval_results" \
        --name "fold_${FOLD}_hsi_rgb" \
        --exist-ok \
        --verbose

    echo ""
    echo "--------------------------------------------"
    echo "  Evaluating Fold ${FOLD} on Official HSI-PCA test"
    echo "--------------------------------------------"

    python3 "${YOLOV5_DIR}/val.py" \
        --weights "${BEST_WEIGHTS}" \
        --data "${DATA_DIR}/test_official_hsi_pca_v2.yaml" \
        --img 640 \
        --batch-size 32 \
        --single-cls \
        --project "${DATA_DIR}/eval_results" \
        --name "fold_${FOLD}_hsi_pca" \
        --exist-ok \
        --verbose
done

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  Results in: ${DATA_DIR}/eval_results/"
echo "============================================"
echo ""
echo "Paper reference (Table I & II lower bounds):"
echo "  HSI-RGB: 59.03 ± 0.81 mAP@0.5"
echo "  HSI-PCA: 23.27 ± 0.05 mAP@0.5"
