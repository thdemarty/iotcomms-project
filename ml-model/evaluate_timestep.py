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

# Import the architecture and loader from your resnet_timestep script
from resnet_timestep import ResNet1D, load_and_prepare_data 

# ==========================================
# 1. Configuration & Setup
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))

DEVICE = torch.device(os.getenv("DEVICE", "cpu"))

# ==========================================
# 2. Evaluation Logic
# ==========================================

def evaluate_timestep_model(scenario, method):
    # 1. Load Data using the Timestep-aware loader
    _, test_loader, num_classes = load_and_prepare_data(scenario, method)
    
    if scenario == 1:
        class_names = ['Bridge', 'Forest', 'Garden', 'Lake', 'River']
    else:
        class_names = [f'Node {i}' for i in range(5)]

    # 2. Load the specific Timestep model weights
    model_filename = f"resnet_timestep_S{scenario}_M{method}.pth"
    model_path = os.path.join(script_dir, "saves", model_filename)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find weights at {model_path}")
        return

    # Initialize model (2 channels: Timestep + RSSI)
    model = ResNet1D(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    # 3. Inference
    print(f"Evaluating Timestep Model: Scenario {scenario}, Method {method}...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 4. Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*45)
    print(f"TIMESTEP MODEL RESULTS: S{scenario} | M{method}")
    print("="*45)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print("="*45 + "\n")

    # 5. Plotting
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # Different color to distinguish from RSSI-only
                xticklabels=class_names[:num_classes], 
                yticklabels=class_names[:num_classes])
    
    plt.title(f'Confusion Matrix (ResNet + Timestep) - S{scenario}, M{method}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = os.path.join(script_dir, f"CM_Timestep_S{scenario}_M{method}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Timestep Confusion Matrix saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--method", type=int, required=True)
    args = parser.parse_args()
    
    evaluate_timestep_model(args.scenario, args.method)