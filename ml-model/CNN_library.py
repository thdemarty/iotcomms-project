import pandas as pd
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

def plot_confusion(cm, labels, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()

def plot_training_curve(epoch_data):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax[0].plot(epoch_data["epoch"].to_numpy(), epoch_data["loss"].to_numpy(), color="red")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)

    # Accuracy plot
    ax[1].plot(epoch_data["epoch"].to_numpy(), epoch_data["node_acc"].to_numpy(), label="Node Accuracy")
    ax[1].plot(epoch_data["epoch"].to_numpy(), epoch_data["env_acc"].to_numpy(), label="Environment Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300)
    plt.show()


class Method(Enum):
    MIX4TRAIN = 0
    MIX4TEST = 1
    VALIDATE = 2

class Mode(Enum):
    NODE = 0
    ENV = 1

class BLEDataset(Dataset):
    def __init__(self, csv_path, method=None, mode=None, filter_id=None, window_size=100, overlap=0.5):
        """
        csv_path : path to csv
        method : see in class Method. MIXALL, MIX4TRAIN, MIX4TEST
        mode : filter for node or environment
        filter_id : which environment or node to split off in scenario 1
        window_size : samples per frame
        overlap : how much the frames overlap with their neighbours
        """
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))

        file_pattern = r"/ml-model/.*\.py$"
        df = pd.read_csv(re.sub(file_pattern, "/data/" + csv_path, __file__))

        # Encode labels
        self.device_encoder = LabelEncoder()
        self.env_encoder = LabelEncoder()

        df["device_label"] = self.device_encoder.fit_transform(df["NodeID"])
        df["env_label"] = self.env_encoder.fit_transform(df["Environment"])

        if mode == Mode.NODE:
            primary_label = "device_label"
            secondary_label = "env_label"
        elif mode == Mode.ENV:
            primary_label = "env_label"
            secondary_label = "device_label"

        samples = []
        device_labels = []
        env_labels = []

        if method == Method.MIX4TRAIN:
            # if mode_id not given or not in dataset : # mix all belonging to node/environment X (randomise, split 75/25 later)
            # if mode_id is given : # mix 4 belonging to node/environment X (use one environment/node belonging to node/environment X for training)
            for id in df[primary_label].unique():
                detached_df = df[df[primary_label] == id]
                detached_df = detached_df[detached_df[secondary_label] != filter_id]

                # Sort by time if timestamp exists
                if "Timestamp" in detached_df.columns:
                    detached_df = detached_df.sort_values("Timestamp")

                signal = detached_df[["RSSI", "LQI"]].values
                # differentiate
                signal = np.diff(signal, axis=0)
                signal = np.vstack([signal[0], signal])
                # normalize
                signal = (signal - np.min(signal, axis=0)) / (np.max(signal, axis=0) - np.min(signal, axis=0))

                env = detached_df["env_label"].values
                dev = detached_df["device_label"].values

                windows = self.create_windows(signal)
                env_windows = self.create_windows(env)
                dev_windows = self.create_windows(dev)

                # print(windows, env_windows, dev_windows)

                for i in range(len(windows)):
                    samples.append(windows[i])
                    env_labels.append(env_windows[i][0])   # environment of window
                    device_labels.append(dev_windows[i][0])
        
        elif method == Method.MIX4TEST:
            for id in df[primary_label].unique():
                detached_df = df[df[primary_label] == id]
                detached_df = detached_df[detached_df[secondary_label] == filter_id]

                # Sort by time if timestamp exists
                if "Timestamp" in detached_df.columns:
                    detached_df = detached_df.sort_values("Timestamp")

                signal = detached_df[["RSSI", "LQI"]].values
                # differentiate
                signal = np.diff(signal, axis=0)
                signal = np.vstack([signal[0], signal])
                # normalize
                signal = (signal - np.min(signal, axis=0)) / (np.max(signal, axis=0) - np.min(signal, axis=0))

                env = detached_df["env_label"].values
                dev = detached_df["device_label"].values

                windows = self.create_windows(signal)
                env_windows = self.create_windows(env)
                dev_windows = self.create_windows(dev)

                # print(windows, env_windows, dev_windows)

                for i in range(len(windows)):
                    samples.append(windows[i])
                    env_labels.append(env_windows[i][0])   # environment of window
                    device_labels.append(dev_windows[i][0])

        elif method == Method.VALIDATE:
            # Sort by time if timestamp exists
            if "Timestamp" in df.columns:
                df = df.sort_values("Timestamp")

            signal = df[["RSSI", "LQI"]].values
            # differentiate
            signal = np.diff(signal, axis=0)
            signal = np.vstack([signal[0], signal])
            # normalize
            signal = (signal - np.min(signal, axis=0)) / (np.max(signal, axis=0) - np.min(signal, axis=0))

            env = df["env_label"].values
            dev = df["device_label"].values

            windows = self.create_windows(signal)
            env_windows = self.create_windows(env)
            dev_windows = self.create_windows(dev)

            # print(windows, env_windows, dev_windows)

            for i in range(len(windows)):
                samples.append(windows[i])
                env_labels.append(env_windows[i][0])   # environment of window
                device_labels.append(dev_windows[i][0])

        else:
            print(f"ERROR! Method for BLEDataset either not given or wrong: {{{method}}}")
            exit()

        self.X = torch.tensor(np.array(samples), dtype=torch.float32)
        self.y_device = torch.tensor(device_labels, dtype=torch.long)
        self.y_env = torch.tensor(env_labels, dtype=torch.long)

    def create_windows(self, data):
        windows = []
        # print(0, len(data), self.window_size, self.step_size)
        for start in range(0, len(data) - self.window_size + 1, self.step_size):
            windows.append(data[start:start + self.window_size])
        return np.array(windows)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Shape: (channels, time)
        return self.X[idx].T, self.y_device[idx], self.y_env[idx]

class BLECNN(nn.Module):
    def __init__(self, num_devices=5, num_envs=5, device="cpu"):
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

        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.device_head = nn.Linear(128, num_devices)
        self.env_head = nn.Linear(128, num_envs)

        self.training_curve = pd.DataFrame(columns=["epoch", "loss", "node_acc", "env_acc"])

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
    
    def start_training(self, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            dev_acc, env_acc = self.evaluate(test_loader)

            self.training_curve.loc[len(self.training_curve)] = [epoch + 1, train_loss, dev_acc, env_acc]

            print(
                f"Epoch {epoch+1:02d} | "
                f"Loss: {train_loss:.4f} | "
                f"Device Acc: {dev_acc:.3f} | "
                f"Env Acc: {env_acc:.3f}"
            )
    
    def get_training_curve(self):
        return self.training_curve

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
    
    def collect_predictions(self, loader):
        self.eval()

        dev_true, dev_pred = [], []
        env_true, env_pred = [], []

        with torch.no_grad():
            for x, y_dev, y_env in loader:
                x = x.to(self.device)

                out_dev, out_env = self(x)

                dev_pred.extend(out_dev.argmax(1).cpu().numpy())
                env_pred.extend(out_env.argmax(1).cpu().numpy())

                dev_true.extend(y_dev.numpy())
                env_true.extend(y_env.numpy())

        return (
            np.array(dev_true),
            np.array(dev_pred),
            np.array(env_true),
            np.array(env_pred),
        )


class CNN:
    def __init__(self, model_paths, device_weights=None, env_weights=None,
                 num_devices=5, num_envs=5, device="cpu"):
        """
        model_paths: list of paths to saved BLECNN state_dicts
        device_weights: trust weights for device predictions (per model)
        env_weights: trust weights for env predictions (per model)
        num_devices: number of device classes
        num_envs: number of environment classes
        device: 'cpu' or 'cuda'
        """
        self.device = device
        self.models = []
        for path in model_paths:
            model = BLECNN(num_devices=num_devices, num_envs=num_envs, device=device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            self.models.append(model)

        n = len(self.models)

        # Set weights (default equal)
        if device_weights is None:
            device_weights = [1.0] * n
        if env_weights is None:
            env_weights = [1.0] * n

        self.device_weights = np.array(device_weights, dtype=float)
        self.env_weights = np.array(env_weights, dtype=float)

        # Normalize to sum=1
        self.device_weights /= self.device_weights.sum()
        self.env_weights /= self.env_weights.sum()

    def predict(self, x):
        """
        x: input tensor (batch_size, channels, seq_len)
        Returns:
            device_pred, env_pred: predicted class indices
            device_probs, env_probs: weighted probability tensors
        """
        x = x.to(self.device)

        weighted_device = None
        weighted_env = None

        for idx, model in enumerate(self.models):
            dev_out, env_out = model(x)  # logits
            dev_prob = F.softmax(dev_out, dim=1)
            env_prob = F.softmax(env_out, dim=1)

            # Weighted sum per metric
            if weighted_device is None:
                weighted_device = self.device_weights[idx] * dev_prob
                weighted_env = self.env_weights[idx] * env_prob
            else:
                weighted_device += self.device_weights[idx] * dev_prob
                weighted_env += self.env_weights[idx] * env_prob

        device_pred = weighted_device.argmax(dim=1)
        env_pred = weighted_env.argmax(dim=1)

        return device_pred, env_pred, weighted_device, weighted_env

    def evaluate(self, loader):
        """Evaluate the ensemble on a dataloader."""
        correct_dev = 0
        correct_env = 0
        total = 0

        with torch.no_grad():
            for x, y_dev, y_env in loader:
                x, y_dev, y_env = x.to(self.device), y_dev.to(self.device), y_env.to(self.device)
                dev_pred, env_pred, _, _ = self.predict(x)

                correct_dev += (dev_pred == y_dev).sum().item()
                correct_env += (env_pred == y_env).sum().item()
                total += y_dev.size(0)

        return correct_dev / total, correct_env / total

    def collect_predictions(self, loader):
        """Collect true and predicted labels for all samples."""
        dev_true, dev_pred = [], []
        env_true, env_pred = [], []

        with torch.no_grad():
            for x, y_dev, y_env in loader:
                x = x.to(self.device)
                dp, ep, _, _ = self.predict(x)

                dev_pred.extend(dp.cpu().numpy())
                env_pred.extend(ep.cpu().numpy())

                dev_true.extend(y_dev.numpy())
                env_true.extend(y_env.numpy())

        return np.array(dev_true), np.array(dev_pred), np.array(env_true), np.array(env_pred)




if __name__=="__main__":
    ### 
    print("This is a custom library for training and testing CNNs enabled by pytorch and sklearn.\n"
          "Nothing happens by running it directly.")