import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# ==========================================
# 1. Load Configurations from .env
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))

FILTER_ID = int(os.getenv("FILTER_ID", 4))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.25))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
DEVICE = os.getenv("DEVICE", "cpu")
NUM_DEVICES = int(os.getenv("NUM_DEVICES", 5))
NUM_ENVS = int(os.getenv("NUM_ENVS", 5))
EPOCHS = int(os.getenv("EPOCHS", 20))
FRAMESIZE = int(os.getenv("FRAMESIZE", 100))
OVERLAP = float(os.getenv("OVERLAP", 0.5))

# Fallback to cuda if available and set to cpu by mistake
if DEVICE == "cpu" and torch.cuda.is_available():
    print("CUDA detected! Consider changing DEVICE=cuda in your .env for faster training.")

# ==========================================
# 2. 1D ResNet Architecture
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
        
        # Initial Convolution: (Batch, 1, FRAMESIZE) -> (Batch, 16, FRAMESIZE/2)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Blocks
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        # Classification Head
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
# 3. Data Processing & Framing
# ==========================================
def create_frames(df, target_col):
    """Applies sliding window over RSSI values."""
    step = int(FRAMESIZE * (1 - OVERLAP))
    frames, labels = [], []
    
    # Group by env and node so windows don't cross over different physical recordings
    for (env, node), group in df.groupby(['env_id', 'node_id']):
        rssi_vals = group['rssi'].values
        target_vals = group[target_col].values
        
        for i in range(0, len(rssi_vals) - FRAMESIZE + 1, step):
            frames.append(rssi_vals[i : i + FRAMESIZE])
            labels.append(target_vals[i]) # Target is identical across this specific frame
            
    return np.array(frames), np.array(labels)

def load_and_prepare_data(scenario, method):
    csv_path = os.path.join(script_dir, '../data/dataset.csv')
    print(f"Loading data from {os.path.abspath(csv_path)}...")
    df = pd.read_csv(csv_path)
    
    # 1. SCENARIO CONFIGURATION
    if scenario == 1:
        print("Scenario I: Classifying Environments")
        target_col = 'env_id'
        num_classes = NUM_ENVS
        filter_col = 'node_id' # Leave a NODE out to test unseen data
    else:
        print("Scenario II: Classifying Nodes")
        target_col = 'node_id'
        num_classes = NUM_DEVICES
        filter_col = 'env_id'  # Leave an ENVIRONMENT out to test unseen data

    # 2. METHOD CONFIGURATION
    if method == 1:
        print(f"Method 1: Random {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split from same dataset")
        X, y = create_frames(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
    elif method == 2:
        print(f"Method 2: Unseen data. Leaving out {filter_col} == {FILTER_ID} for testing")
        df_train = df[df[filter_col] != FILTER_ID]
        df_test = df[df[filter_col] == FILTER_ID]
        
        X_train, y_train = create_frames(df_train, target_col)
        X_test, y_test = create_frames(df_test, target_col)
        
        if len(X_test) == 0:
            raise ValueError(f"Error: No testing data found for {filter_col} == {FILTER_ID}. Check your FILTER_ID in .env")

    print(f"Generated {len(X_train)} Training frames and {len(X_test)} Testing frames.\n")

    # 3. Convert to PyTorch Tensors: Shape -> (Batch, Channels=1, Framesize)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, num_classes

# ==========================================
# 4. Training Loop
# ==========================================
def train_model(scenario, method):
    train_loader, test_loader, num_classes = load_and_prepare_data(scenario, method)
    device = torch.device(DEVICE)
    
    model = ResNet1D(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting Training on {DEVICE.upper()} for {EPOCHS} Epochs")
    print("-" * 50)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # Evaluation Phase
        model.eval()
        test_loss, t_correct, t_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                t_total += labels.size(0)
                t_correct += predicted.eq(labels).sum().item()
                
        test_acc = 100. * t_correct / t_total
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss/total:.4f} | Train Acc: {train_acc:.2f}% || Test Loss: {test_loss/t_total:.4f} | Test Acc: {test_acc:.2f}%")

    # Save Model Weights
    save_dir = os.path.join(script_dir, "saves")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"resnet_scenario{scenario}_method{method}.pth")
    torch.save(model.state_dict(), save_path)
    print("-" * 50)
    print(f"✅ Training Complete. Model saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D ResNet on RSSI Data")
    parser.add_argument("--scenario", type=int, required=True, choices=[1, 2], help="1: Predict Environment, 2: Predict Node")
    parser.add_argument("--method", type=int, required=True, choices=[1, 2], help="1: Standard Split, 2: Leave-one-out Split")
    args = parser.parse_args()
    
    train_model(args.scenario, args.method)
