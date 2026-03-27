#!/bin/bash

# Create saves directory if it doesn't exist
mkdir -p saves

# Output file for results
RESULT_FILE="results.txt"
echo "RESULTS" > $RESULT_FILE
echo "Generated on: $(date)" >> $RESULT_FILE
echo "------------------------------------------------" >> $RESULT_FILE

# Function to run training and evaluation
run_experiment() {
    MODEL=$1 # "CNN" or "ResNet"
    SCENARIO=$2
    METHOD=$3

    if [ "$MODEL" = "CNN" ]; then
        SCRIPT="train_cnn.py"
    elif [ "$MODEL" = "ResNet" ]; then
        SCRIPT="train_resnet.py"
    fi

    echo "Running: $MODEL | Scenario $SCENARIO | Method $METHOD"
    python "$SCRIPT" --scenario "$SCENARIO " --method "$METHOD" | tee -a $RESULT_FILE
    echo "------------------------------------------------" >> $RESULT_FILE
}

# --- EXECUTION LOOP ---
# Scenarios: 1 (Env), 2 (Node)
# Methods: 1 (Random), 2 (Leave-one-out)

for model in "CNN" "ResNet"; do
    for s in 1 2; do
        for m in 1 2; do
            # Run Standard RSSI
            run_experiment $model $s $m "false"
        done
    done
done

echo "All experiments completed!"
echo "Check '$RESULT_FILE' for results."
