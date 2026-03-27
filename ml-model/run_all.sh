#!/bin/bash

# Create saves directory if it doesn't exist
mkdir -p saves

# Output file for results
RESULT_FILE="results.txt"
echo "RESULTS" | tee $RESULT_FILE
echo "Generated on: $(date)" | tee -a $RESULT_FILE
echo "------------------------------------------------" | tee -a $RESULT_FILE

# Function to run training and evaluation
run_experiment() {
    MODEL=$1 # "CNN" or "ResNet"
    SCENARIO=$2
    METHOD=$3
    ID=$4

    if [ "$MODEL" = "CNN" ]; then
        SCRIPT="train_cnn.py"
    elif [ "$MODEL" = "ResNet" ]; then
        SCRIPT="train_resnet.py"
    fi

    if [ "$ID" = '' ]; then
        echo "Running: $MODEL | Scenario $SCENARIO | Method $METHOD" | tee -a $RESULT_FILE
        python "$SCRIPT" --scenario "$SCENARIO" --method "$METHOD" | tee -a $RESULT_FILE
    else
        echo "Running: $MODEL | Scenario $SCENARIO | Method $METHOD | ID $ID" | tee -a $RESULT_FILE
        python "$SCRIPT" --scenario "$SCENARIO" --method "$METHOD" --id "$ID" | tee -a $RESULT_FILE
    fi
    echo "------------------------------------------------" | tee -a $RESULT_FILE
}

# --- EXECUTION LOOP ---
# Scenarios: 1 (Env), 2 (Node)
# Methods: 1 (Random), 2 (Leave-one-out)

for model in "CNN" "ResNet"; do
    for s in 1 2; do
        run_experiment $model $s 1
        for ID in {0..4}; do
            run_experiment $model $s 2 $ID
        done
    done
done

echo "All experiments completed!"
echo "Check '$RESULT_FILE' for results."
