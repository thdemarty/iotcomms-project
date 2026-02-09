import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re

class BLEDataset(Dataset):
    def __init__(self, csv_path, window_size=100, overlap=0.5):
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))

        file_pattern = r"/ml-model/.*\.py$"
        df = pd.read_csv(re.sub(file_pattern, "/data/" + csv_path, __file__))

        # Encode labels
        self.device_encoder = LabelEncoder()
        self.env_encoder = LabelEncoder()

        df["device_label"] = self.device_encoder.fit_transform(df["NodeID"])
        df["env_label"] = self.env_encoder.fit_transform(df["Environment"])

        samples = []
        device_labels = []
        env_labels = []

        # Create windows PER DEVICE (but mixed environments)
        for device_id in df["device_label"].unique():
            device_df = df[df["device_label"] == device_id]

            # Sort by time if timestamp exists
            if "Timestamp" in device_df.columns:
                device_df = device_df.sort_values("Timestamp")

            signal = device_df[["RSSI", "LQI"]].values
            env = device_df["env_label"].values
            dev = device_df["device_label"].values

            windows = self.create_windows(signal)
            env_windows = self.create_windows(env)
            dev_windows = self.create_windows(dev)

            print(windows, env_windows, dev_windows)

            for i in range(len(windows)):
                samples.append(windows[i])
                env_labels.append(env_windows[i][0])   # environment of window
                device_labels.append(dev_windows[i][0])

        self.X = torch.tensor(np.array(samples), dtype=torch.float32)
        self.y_device = torch.tensor(device_labels, dtype=torch.long)
        self.y_env = torch.tensor(env_labels, dtype=torch.long)

    def create_windows(self, data):
        windows = []
        print(0, len(data), self.window_size, self.step_size)
        for start in range(0, len(data) - self.window_size + 1, self.step_size):
            windows.append(data[start:start + self.window_size])
        return np.array(windows)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Shape: (channels, time)
        return self.X[idx].T, self.y_device[idx], self.y_env[idx]

class BLECNN(nn.Module):
    def __init__(self, num_devices=5, num_envs=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.device_head = nn.Linear(128, num_devices)
        self.env_head = nn.Linear(128, num_envs)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)

        device_out = self.device_head(x)
        env_out = self.env_head(x)

        return device_out, env_out
    
    def train_epoch(self, loader):
        self.train()
        total_loss = 0

        for x, y_dev, y_env in loader:
            x, y_dev, y_env = x.to(self.device), y_dev.to(self.device), y_env.to(self.device)

            self.optimizer.zero_grad()

            dev_out, env_out = self(x)

            loss_dev = self.criterion(dev_out, y_dev)
            loss_env = self.criterion(env_out, y_env)

            loss = loss_dev + loss_env
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader):
        self.eval()
        correct_dev = 0
        correct_env = 0
        total = 0

        with torch.no_grad():
            for x, y_dev, y_env in loader:
                x, y_dev, y_env = x.to(self.device), y_dev.to(self.device), y_env.to(self.device)

                dev_out, env_out = self(x)

                _, dev_pred = torch.max(dev_out, 1)
                _, env_pred = torch.max(env_out, 1)

                correct_dev += (dev_pred == y_dev).sum().item()
                correct_env += (env_pred == y_env).sum().item()
                total += y_dev.size(0)

        return correct_dev / total, correct_env / total


if __name__=="__main__":
    ### Train / test split
    dataset = BLEDataset("test_ts.csv")

    indices = np.arange(len(dataset))
    print(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42, shuffle=True)
    print(train_idx, test_idx)

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    ### Training loop (CPU only)
    device = torch.device("cpu")
    model = BLECNN(num_devices=5, num_envs=5).to(device)

    ### Run training
    epochs = 20

    for epoch in tqdm(range(epochs)):
        train_loss = model.train_epoch(train_loader)
        dev_acc, env_acc = model.evaluate(test_loader)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Device Acc: {dev_acc:.3f} | "
            f"Env Acc: {env_acc:.3f}"
        )
