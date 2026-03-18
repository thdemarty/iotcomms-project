import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from query_script import dataframe_query
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIGURATION
# =========================
FRAME_SIZE = 100
OVERLAP = 0.5
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

file_pattern = r"/ml-model/.*\.py$"

# =========================
# DATASET CLASS
# =========================
class RSSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# FRAME GENERATION
# =========================
def create_frames(signal, frame_size, overlap):
    step = int(frame_size * (1 - overlap))
    frames = []

    for start in range(0, len(signal) - frame_size + 1, step):
        frames.append(signal[start:start + frame_size])

    return np.array(frames)


def prepare_frames(df, label):
    """
    df: dataframe returned by query_dataset (columns: receiver_id, timestep, rssi)
    label: int or str; known sender_id or environment_id
    """
    grouped_frames = []

    for receiver_id, group in df.groupby("receiver_id"):
        rssi = group["rssi"].values
        timestep = group["timestep"].values

        # create overlapping frames
        rssi_frames = create_frames(rssi, FRAME_SIZE, OVERLAP)
        timestep_frames = create_frames(timestep, FRAME_SIZE, OVERLAP)

        # combine RSSI + timestep into 2 channels
        for r, t in zip(rssi_frames, timestep_frames):
            frame = np.stack([r, t], axis=0)  # shape (2, FRAME_SIZE)
            grouped_frames.append({
                "frame": frame,
                "label": label
            })
    return grouped_frames


# =========================
# SCENARIO + METHOD SPLITS
# =========================
def split_data(frames_data):
    X = np.array([d["frame"] for d in frames_data])
    labels = np.array([d["label"] for d in frames_data])
    
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    if METHOD == 1:
        # Random 75/25 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels_enc, test_size=0.25, stratify=labels_enc
        )
    else:
        # Leave-one-environment-out
        envs = np.array([d["env"] for d in frames_data])
        unique_envs = np.unique(envs)
        test_env = unique_envs[ID]  # ID env for testing

        train_idx = envs != test_env
        test_idx = envs == test_env

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels_enc[train_idx], labels_enc[test_idx]

    return X_train, X_test, y_train, y_test, le


# =========================
# CNN MODEL
# =========================
class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),

            nn.Linear((input_size // 4) * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# TRAINING WITH EPOCH LOGGING
# =========================
def train_model(X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = RSSIDataset(X_train, y_train)
    test_ds = RSSIDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = CNN1D(FRAME_SIZE, len(np.unique(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    epoch_data = {"epoch": [], "loss": [], "accuracy": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # evaluate test accuracy
        acc = evaluate(model, test_loader, device, verbose=False)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Test Acc: {acc:.4f}")
        epoch_data["epoch"].append(epoch + 1)
        epoch_data["loss"].append(total_loss)
        epoch_data["accuracy"].append(acc)

    # plot training curve
    plot_training_curve(epoch_data)

    # final confusion matrix
    y_true, y_pred = get_predictions(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    labels = list(np.unique(y_true))
    plot_confusion(cm, labels, f"Confusion_s{SCENARIO}_m{METHOD}")

    return model


# =========================
# EVALUATION
# =========================
def evaluate(model, loader, device, verbose=True):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb).argmax(dim=1)

            correct += (preds == yb).sum().item()
            total += len(yb)
    acc = correct / total
    if acc==None:
        acc = 0.0
    if verbose:
        print(f"Test Accuracy: {acc:.4f}")
    else:
        return acc


# =========================
# GET PREDICTIONS FOR CONFUSION MATRIX
# =========================
def get_predictions(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb).argmax(dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return np.array(y_true), np.array(y_pred)


# =========================
# PLOTTING FUNCTIONS
# =========================
def plot_confusion(cm, labels, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


def plot_training_curve(epoch_data):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epoch_data["epoch"], epoch_data["loss"], color="red")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)
    ax[1].plot(epoch_data["epoch"], epoch_data["accuracy"], label="Test Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300)
    plt.show()


# =========================
# SAVE WEIGHTS
# =========================
def save_model(model):
    filename = f"cnn_s{SCENARIO}_m{METHOD}_f{FRAME_SIZE}_o{int(OVERLAP*100)}.pt"
    path = os.path.join(WEIGHTS_DIR, filename)

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    # setup arg parser
    parser = argparse.ArgumentParser(prog='train_cnn', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=int, default=1, help='Select the scenario. For help read the readme in ml-model.')
    parser.add_argument('--method', type=int, default=1, help='Select the method. For help read the readme in ml-model.')
    parser.add_argument('--id', type=int, default=0, help='Which id will be used for testing in method 2. For help read the readme in ml-model.')
    args = parser.parse_args()
    
    # read args
    SCENARIO = args.scenario    # 1 = environment, 2 = sensor node
    METHOD = args.method        # 1 = random split, 2 = leave-one-env-out
    ID = args.id

    scenario_description = ["environment", "sending node"]
    method_description = ["random split 75/25", f"{scenario_description[SCENARIO]} id {ID} for testing"]
    print(
        f"""
        Starting the training of the CNN.
        ---------------------------------------
        Scenario {SCENARIO} : Looking for {scenario_description[SCENARIO]}
        Method {METHOD}     : Using {method_description[METHOD]}

        """
        )

    dataset_path = re.sub(file_pattern, "/data/dataset.csv", __file__)

    sender_ids = [0, 1, 2, 3, 4]
    env_ids = [0, 1, 2, 3, 4]

    all_frames = []

    # Example loop over senders and environments
    if SCENARIO == 1:
        # Goal: recognize environment
        for env_id in env_ids:
            df = dataframe_query(dataset_path, sender_ids, [env_id])
            frames = prepare_frames(df, label=env_id)
            all_frames.extend(frames)
    else:
        # Goal: recognize sender
        for sender_id in sender_ids:
            df = dataframe_query(dataset_path, [sender_id], env_ids)
            frames = prepare_frames(df, label=sender_id)
            all_frames.extend(frames)

    X_train, X_test, y_train, y_test, le = split_data(all_frames)
    model = train_model(X_train, y_train, X_test, y_test)
    save_model(model)