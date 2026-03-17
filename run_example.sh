#!/bin/bash

# ActiveCQ Example: Run active learning on simulation data with discrete treatment (CATE task)
# This script demonstrates a simple round of active learning + evaluation.

JOB_DIR="experiments/example"
NUM_TRIALS=3
ACQ_SIZE=20
WARM_SIZE=50
MAX_ACQ=10
NUM_EXAMPLES=500
TASK_TYPE="cate"
TREATMENT_TYPE="discrete"

LOG_DIR="$JOB_DIR/logs"
mkdir -p "$LOG_DIR"

METHODS=("random" "acqe")

for method in "${METHODS[@]}"; do
    if [ "$method" == "acqe" ]; then
        ACQ_FUNC="acqe"
        STRATEGY="VR"
        CDE="CME"
        BATCH="B"
        METHOD_LABEL="VR_CME_B"
    else
        ACQ_FUNC="random"
        STRATEGY="IG"
        CDE="CME"
        BATCH="B"
        METHOD_LABEL="random"
    fi

    echo "===== Running active learning: $METHOD_LABEL ====="
    python src/application/main.py \
        active-learning \
            --job-dir "$JOB_DIR" \
            --num-trials $NUM_TRIALS \
            --acq-size $ACQ_SIZE \
            --warm-start-size $WARM_SIZE \
            --max-acquisitions $MAX_ACQ \
            --acquisition-function "$ACQ_FUNC" \
            --adaptive-strategy "$STRATEGY" \
            --cde-estimator "$CDE" \
            --batch-aware "$BATCH" \
        simulation \
            --num-examples $NUM_EXAMPLES \
            --task_type "$TASK_TYPE" \
            --treatment_type "$TREATMENT_TYPE" \
        imp \
            --learning-rate 0.05 \
            --gp_epochs 500 \
            --learn_cme True \
    2>&1 | tee "$LOG_DIR/${METHOD_LABEL}_train.log"

    echo "===== Evaluating: $METHOD_LABEL ====="
    python src/application/main.py \
        evaluate \
            --experiment-dir "$JOB_DIR/active_learning/$METHOD_LABEL" \
            --output-dir "$JOB_DIR/results" \
        amse \
    2>&1 | tee "$LOG_DIR/${METHOD_LABEL}_eval.log"
done

echo "===== Plotting convergence ====="
python src/application/main.py \
    evaluate \
        --experiment-dir "$JOB_DIR/results" \
    plot-convergence-in-out \
        --prefix example \
        -m random \
        -m VR_CME_B \
2>&1 | tee "$LOG_DIR/plot.log"

echo "Done! Results saved to $JOB_DIR/results/, logs saved to $LOG_DIR/"
