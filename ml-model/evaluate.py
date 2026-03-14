import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Import the updated architecture and loader from your training script
from resnet import ResNet1D, load_and_prepare_data 

# ==========================================
# 1. Configuration & Setup
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))

DEVICE = torch.device(os.getenv("DEVICE", "cpu"))

# ==========================================
# 2. Evaluation & Plotting
# ==========================================

def evaluate(scenario, method):
    # 1. Load Data and Model using the same logic as Training
    # We use the test_loader from your training script to ensure identical processing
    _, test_loader, num_classes = load_and_prepare_data(scenario, method)
    
    # Define class names for plotting
    if scenario == 1:
        class_names = ['Bridge', 'Forest', 'Garden', 'Lake', 'River']
    else:
        class_names = [f'Node {i}' for i in range(5)]

    model_path = os.path.join(script_dir, "saves", f"resnet_scenario{scenario}_method{method}.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained weights at {model_path}")
        return

    # Initialize model with the NEW architecture (matching 2 channels from resnet.py)
    model = ResNet1D(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    # 2. Run Inference
    print(f"Evaluating Scenario {scenario}, Method {method}...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 3. Calculate Accuracy and Precision
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*40)
    print(f"RESULTS: Scenario {scenario} | Method {method}")
    print("="*40)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print("="*40 + "\n")

    # 4. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names[:num_classes], 
                yticklabels=class_names[:num_classes])
    
    plt.title(f'Confusion Matrix (ResNet) - Scenario {scenario}, Method {method}')
    plt.ylabel('Actual (True Data)')
    plt.xlabel('Predicted (Model Output)')
    plt.tight_layout()
    
    # Save Image
    save_path = os.path.join(script_dir, f"confusion_matrix_S{scenario}_M{method}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Confusion Matrix saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--method", type=int, required=True)
    args = parser.parse_args()
    
    evaluate(args.scenario, args.method)
