import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from dotenv import load_dotenv

# ==========================================
# 1. Configuration & Setup
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))
DEVICE = torch.device(os.getenv("DEVICE", "cpu"))

def run_eval(scenario, method, use_timestep):
    # 2. Dynamic Import based on the flag
    if use_timestep:
        from resnet_timestep import ResNet1D, load_and_prepare_data
        model_file = f"resnet_timestep_S{scenario}_M{method}.pth"
        title_suffix = "(with Timestep)"
        save_suffix = "Timestep"
        color_map = "Greens"
    else:
        from resnet import ResNet1D, load_and_prepare_data
        model_file = f"resnet_S{scenario}_M{method}.pth"
        title_suffix = "(RSSI Only)"
        save_suffix = "RSSI"
        color_map = "Blues"

    print(f"\n--- Evaluating ResNet {title_suffix} | Scenario {scenario} | Method {method} ---")

    # 3. Load Data & Model
    _, test_loader, num_classes = load_and_prepare_data(scenario, method)
    
    model_path = os.path.join(script_dir, "saves", model_file)
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at {model_path}")
        return

    model = ResNet1D(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []
    class_names = ['Bridge', 'Forest', 'Garden', 'Lake', 'River'] if scenario == 1 else [f'Node {i}' for i in range(5)]

    # 4. Inference
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # 5. Metrics (Thomas's F-score requirement)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("-" * 40)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"F-Score:   {f1 * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=class_names[:num_classes]))

    # 6. Plotting Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color_map, 
                xticklabels=class_names[:num_classes], 
                yticklabels=class_names[:num_classes])
    
    plt.title(f'ResNet Confusion Matrix {title_suffix}\nScenario {scenario} | Method {method}')
    plt.ylabel('Actual (True Data)')
    plt.xlabel('Predicted (Model Output)')
    plt.tight_layout()
    
    save_path = os.path.join(script_dir, f"CM_S{scenario}_M{method}_{save_suffix}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Confusion Matrix saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet Models")
    parser.add_argument("--scenario", type=int, required=True, choices=[1, 2])
    parser.add_argument("--method", type=int, required=True, choices=[1, 2])
    parser.add_argument("--timestep", action="store_true", help="Use this flag for resnet_timestep models")
    
    args = parser.parse_args()
    run_eval(args.scenario, args.method, args.timestep)