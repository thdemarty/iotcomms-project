import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# ==========================================
# 1. Configuration & Setup
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))

FILTER_ID = int(os.getenv("FILTER_ID", 4))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.25))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
FRAMESIZE = int(os.getenv("FRAMESIZE", 100))
OVERLAP = float(os.getenv("OVERLAP", 0.5))
DEVICE = torch.device(os.getenv("DEVICE", "cpu"))

# ==========================================
# 2. ResNet Architecture (Must match exactly)
# ==========================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet1D, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==========================================
# 3. Data Loading (Extracting only Test Data)
# ==========================================
def create_frames(df, target_col):
    step = int(FRAMESIZE * (1 - OVERLAP))
    frames, labels = [], []
    for (env, node), group in df.groupby(['env_id', 'node_id']):
        rssi_vals = group['rssi'].values
        target_vals = group[target_col].values
        for i in range(0, len(rssi_vals) - FRAMESIZE + 1, step):
            frames.append(rssi_vals[i : i + FRAMESIZE])
            labels.append(target_vals[i])
    return np.array(frames), np.array(labels)

def get_test_data(scenario, method):
    csv_path = os.path.join(script_dir, '../data/dataset.csv')
    df = pd.read_csv(csv_path)
    
    if scenario == 1:
        target_col, num_classes, filter_col = 'env_id', 5, 'node_id'
        class_names = ['Bridge', 'Forest', 'Garden', 'Lake', 'River']
    else:
        target_col, num_classes, filter_col = 'node_id', 5, 'env_id'
        class_names = [f'Node {i}' for i in range(5)]

    if method == 1:
        X, y = create_frames(df, target_col)
        _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    elif method == 2:
        df_test = df[df[filter_col] == FILTER_ID]
        X_test, y_test = create_frames(df_test, target_col)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_test_t = np.array(y_test)
    
    return X_test_t, y_test_t, num_classes, class_names

# ==========================================
# 4. Evaluation & Plotting
# ==========================================

def evaluate(scenario, method):
    # 1. Load Data and Model
    X_test, y_true, num_classes, class_names = get_test_data(scenario, method)
    model_path = os.path.join(script_dir, "saves", f"resnet_scenario{scenario}_method{method}.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained weights at {model_path}")
        return

    model = ResNet1D(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. Run Inference
    print(f"Evaluating Scenario {scenario}, Method {method}...")
    with torch.no_grad():
        outputs = model(X_test)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()

    # 3. Calculate Accuracy and Precision
    accuracy = accuracy_score(y_true, y_pred)
    # Using weighted precision to handle any minor class imbalances
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    # Presenting dimensions clearly as requested
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
