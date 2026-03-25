#!/bin/bash

# Create saves directory if it doesn't exist
mkdir -p saves

# Output file for results
RESULT_FILE="final_results.txt"
echo "RESNET PROJECT - FINAL RESULTS LOG" > $RESULT_FILE
echo "Generated on: $(date)" >> $RESULT_FILE
echo "------------------------------------------------" >> $RESULT_FILE

# Function to run training and evaluation
run_experiment() {
    SCENARIO=$1
    METHOD=$2
    TIMESTEP_FLAG=$3 # "true" or "false"

    if [ "$TIMESTEP_FLAG" = "true" ]; then
        echo "Running: Timestep ResNet | Scenario $SCENARIO | Method $METHOD"
        python resnet_timestep.py --scenario $SCENARIO --method $METHOD
        # Capture evaluation output and append to result file
        python evaluate.py --scenario $SCENARIO --method $METHOD --timestep >> $RESULT_FILE
    else
        echo "Running: Standard ResNet | Scenario $SCENARIO | Method $METHOD"
        python resnet.py --scenario $SCENARIO --method $METHOD
        # Capture evaluation output and append to result file
        python evaluate.py --scenario $SCENARIO --method $METHOD >> $RESULT_FILE
    fi
    echo "------------------------------------------------" >> $RESULT_FILE
}

# --- EXECUTION LOOP ---

# Scenarios: 1 (Env), 2 (Node)
# Methods: 1 (Random), 2 (Leave-one-out)

for s in 1 2; do
    for m in 1 2; do
        # Run Standard RSSI
        run_experiment $s $m "false"
        
        # Run Timestep + RSSI
        run_experiment $s $m "true"
    done
done

echo "✅ All experiments completed!"
echo "Check '$RESULT_FILE' for Accuracy and F-scores."
echo "Check your folder for the .png Confusion Matrices."