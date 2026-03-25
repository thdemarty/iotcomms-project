import os, argparse, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from query_script import dataframe_query

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))
FILTER_ID, TEST_SIZE, RANDOM_STATE, BATCH_SIZE, DEVICE, EPOCHS, FRAMESIZE, OVERLAP = int(os.getenv("FILTER_ID", 4)), float(os.getenv("TEST_SIZE", 0.25)), int(os.getenv("RANDOM_STATE", 42)), int(os.getenv("BATCH_SIZE", 32)), os.getenv("DEVICE", "cpu"), int(os.getenv("EPOCHS", 20)), int(os.getenv("FRAMESIZE", 100)), float(os.getenv("OVERLAP", 0.5))

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels); self.relu = nn.ReLU(inplace=True); self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False); self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet1D, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16); self.relu = nn.ReLU(inplace=True); self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1, self.layer2, self.layer3 = self._make_layer(16, 2, 1), self._make_layer(32, 2, 2), self._make_layer(64, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(64, num_classes)
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for s in [stride] + [1]*(num_blocks-1):
            layers.append(ResidualBlock1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))))), 1))

def create_frames(df, target_col):
    step = int(FRAMESIZE * (1 - OVERLAP))
    frames, labels = [], []
    
    # Check which time column name exists
    time_col = 'timestamp' if 'timestamp' in df.columns else 'timestep'
    
    if time_col not in df.columns:
        raise KeyError(f"Neither 'timestamp' nor 'timestep' found in CSV. Columns are: {df.columns.tolist()}")

    # Extract both columns for the 2-channel input
    features = df[[time_col, 'rssi']].values
    target_vals = df[target_col].values
    
    for i in range(0, len(features) - FRAMESIZE + 1, step):
        # Transpose so shape is (2, FRAMESIZE)
        frames.append(features[i : i + FRAMESIZE].T)
        labels.append(target_vals[i])
            
    return np.array(frames), np.array(labels)
def load_and_prepare_data(scenario, method):
    csv_path = os.path.join(script_dir, '../data/dataset.csv')
    
    # 1. Load the FULL original data first
    df = pd.read_csv(csv_path)
    
    # 2. Map your column names based on your last error log
    actual_node_col = 'sender_id' 
    actual_env_col = 'env_id'
    target_col = actual_env_col if scenario == 1 else actual_node_col

    # 3. Filtering logic (Standard Method 1)
    if method == 1:
        # Just use the whole dataframe
        X, y = create_frames(df, target_col)
        # Ensure we actually have data
        if len(X) == 0:
            raise ValueError(f"No frames created. Check if FRAMESIZE ({FRAMESIZE}) is larger than the data length.")
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    
    # 4. Filtering logic (Leave-one-out Method 2)
    else:
        df_train = df[df[actual_node_col] != FILTER_ID]
        df_test = df[df[actual_node_col] == FILTER_ID]
        
        X_train, y_train = create_frames(df_train, target_col)
        X_test, y_test = create_frames(df_test, target_col)
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(f"Empty split! Train size: {len(X_train)}, Test size: {len(X_test)}. Check if FILTER_ID {FILTER_ID} exists in {actual_node_col}.")

    # 5. Create DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True), \
           DataLoader(test_ds, batch_size=BATCH_SIZE), 5

def train_model(scenario, method):
    train_loader, _, num_classes = load_and_prepare_data(scenario, method)
    model = ResNet1D(num_classes).to(DEVICE); optimizer = optim.Adam(model.parameters(), lr=0.001); criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); criterion(model(inputs), labels).backward(); optimizer.step()
        print(f"Epoch {epoch+1} complete")
    os.makedirs("saves", exist_ok=True); torch.save(model.state_dict(), f"saves/resnet_timestep_S{scenario}_M{method}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--scenario", type=int); parser.add_argument("--method", type=int)
    args = parser.parse_args(); train_model(args.scenario, args.method)