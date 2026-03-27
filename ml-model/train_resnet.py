import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils import shuffle
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
BATCH_SIZE = 32
EPOCHS = 40
LR = 5e-4

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
    label: int or str
    known sender_id or environment_id
    """
    grouped_frames = []
    for receiver_id, group in df.groupby("receiver_id"):
        rssi = group["rssi"].values
        frames = create_frames(rssi, FRAME_SIZE, OVERLAP)
        for f in frames:
            # normalize per frame
            # f = (f - f.min()) / (f.max() - f.min())
            grouped_frames.append({
                "frame": f,
                "label": label
            })
    return grouped_frames


# =========================
# SCENARIO + METHOD SPLITS
# =========================
def split_data(frames_data, test_data):
    X = np.array([d["frame"] for d in frames_data])
    labels = np.array([d["label"] for d in frames_data])
    
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    if METHOD == 1:
        # Random 75/25 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels_enc, test_size=0.25, stratify=labels_enc, random_state=42
        )
    else:
        # Leave-one-out
        X_test = np.array([d["frame"] for d in test_data])
        y_test = np.array([d["label"] for d in test_data])

        # Shuffle training set like train_test_split would do
        X_train, y_train = shuffle(X, labels, random_state=42)

    return X_train, X_test, y_train, y_test, le


# =========================
# ResNet MODEL
# =========================
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
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet1D, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1, self.layer2, self.layer3 = self._make_layer(16, 2, 1), self._make_layer(32, 2, 2), self._make_layer(64, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for s in [stride] + [1]*(num_blocks-1):
            layers.append(ResidualBlock1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.fc(
            torch.flatten(self.avgpool(self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))))), 1))


# =========================
# TRAINING WITH EPOCH LOGGING
# =========================
def train_model(X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    train_ds = RSSIDataset(X_train, y_train)
    test_ds = RSSIDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = ResNet1D(len(np.unique(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    epoch_data = {"epoch": [], "loss": [], "accuracy": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            Xb += 0.01 * torch.randn_like(Xb)   # maybe performance increase for method 2
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
    plot_confusion(cm, labels, f"Confusionmatrix")

    # 5. Metrics (Thomas's F-score requirement)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # define labels
    if SCENARIO == 1:
        # map numeric environment labels to names
        env_map = {0: "garden", 1: "forest", 2: "lake", 3: "river", 4: "bridge"}
        labels = [env_map[i] for i in labels]
    else:
        # Scenario 2: prepend "node " to numeric labels
        labels = ["node " + str(i) for i in labels]
    
    print("-" * 40)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"F-Score:   {f1 * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=labels))

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
    # define labels
    if SCENARIO == 1:
        # map numeric environment labels to names
        env_map = {0: "garden", 1: "forest", 2: "lake", 3: "river", 4: "bridge"}
        labels = [env_map[i] for i in labels]
    else:
        # Scenario 2: prepend "node " to numeric labels
        labels = ["node " + str(i) for i in labels]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if METHOD==2:
        filename = f"resnet_confusionmatrix_s{SCENARIO}_m{METHOD}_f{FRAME_SIZE}_o{int(OVERLAP*100)}_id{ID}.png"
    else:
        filename = f"resnet_confusionmatrix_s{SCENARIO}_m{METHOD}_f{FRAME_SIZE}_o{int(OVERLAP*100)}.png"
    path = re.sub(file_pattern, "/ml-model/saves/" + filename, __file__)
    plt.savefig(path, dpi=300)
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
    if METHOD==2:
        filename = f"resnet_trainingcurve_s{SCENARIO}_m{METHOD}_f{FRAME_SIZE}_o{int(OVERLAP*100)}_id{ID}.png"
    else:
        filename = f"resnet_trainingcurve_s{SCENARIO}_m{METHOD}_f{FRAME_SIZE}_o{int(OVERLAP*100)}.png"
    path = re.sub(file_pattern, "/ml-model/saves/" + filename, __file__)
    plt.savefig(path, dpi=300)
    plt.show()


# =========================
# SAVE WEIGHTS
# =========================
def save_model(model):
    if METHOD==2:
        filename = f"resnet_s{SCENARIO}_m{METHOD}_f{FRAME_SIZE}_o{int(OVERLAP*100)}_id{ID}.pth"
    else:
        filename = f"resnet_s{SCENARIO}_m{METHOD}_f{FRAME_SIZE}_o{int(OVERLAP*100)}.pth"
    path = re.sub(file_pattern, "/ml-model/saves/" + filename, __file__)

    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


if __name__ == "__main__":
    # setup arg parser
    parser = argparse.ArgumentParser(prog='train_resnet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=int, default=1, help='Select the scenario. For help read the readme in ml-model.')
    parser.add_argument('--method', type=int, default=1, help='Select the method. For help read the readme in ml-model.')
    parser.add_argument('--id', type=int, default=0, help='Which id will be used for testing in method 2. For help read the readme in ml-model.')
    
    # parser.add_argument('--frame_size', type=int, default=1, help='')
    # parser.add_argument('--overlap', type=float, default=0.5, help='')
    # parser.add_argument('--batch_size', type=int, default=64, help='')
    # parser.add_argument('--epochs', type=int, default=20, help='')
    # parser.add_argument('--lr', type=float, default=1e-3, help='')
    
    args = parser.parse_args()
    
    # read args
    SCENARIO = args.scenario    # 1 = environment, 2 = sensor node
    METHOD = args.method        # 1 = random split, 2 = leave-one-env-out
    ID = args.id

    # FRAME_SIZE = args.frame_size
    # OVERLAP = args.overlap
    # BATCH_SIZE = args.batch_size
    # EPOCHS = args.epochs
    # LR = args.lr

    print(FRAME_SIZE, OVERLAP, BATCH_SIZE, EPOCHS, LR)
    print(type(FRAME_SIZE), type(OVERLAP), type(BATCH_SIZE), type(EPOCHS), type(LR))

    scenario_description = ["environment", "sending node"]
    method_description = ["random split 75/25", f"{scenario_description[1-(SCENARIO-1)]} id {ID} for testing"]
    print(
        f"""
        Starting the training of the ResNet.
        ---------------------------------------
        Scenario {SCENARIO} : Looking for {scenario_description[SCENARIO-1]}
        Method {METHOD}     : Using {method_description[METHOD-1]}

        """
        )

    dataset_path = re.sub(file_pattern, "/data/dataset.csv", __file__)

    sender_ids = [0, 1, 2, 3, 4]
    env_ids = [0, 1, 2, 3, 4]

    all_frames = []
    test_frames = []

    # Example loop over senders and environments
    if SCENARIO == 1:
        if METHOD == 2:
            sender_ids.remove(ID)
        # Goal: recognize environment
        for env_id in env_ids:
            df = dataframe_query(dataset_path, sender_ids, [env_id])
            frames = prepare_frames(df, label=env_id)
            all_frames.extend(frames)
            if METHOD == 2:
                df = dataframe_query(dataset_path, [ID], [env_id])
                frames = prepare_frames(df, label=env_id)
                test_frames.extend(frames)

    else:
        if METHOD == 2:
            env_ids.remove(ID)
        # Goal: recognize sender
        for sender_id in sender_ids:
            df = dataframe_query(dataset_path, [sender_id], env_ids)
            frames = prepare_frames(df, label=sender_id)
            all_frames.extend(frames)
            if METHOD==2:
                df = dataframe_query(dataset_path, [sender_id], [ID])
                frames = prepare_frames(df, label=sender_id)
                test_frames.extend(frames)

    X_train, X_test, y_train, y_test, le = split_data(all_frames, test_frames)
    model = train_model(X_train, y_train, X_test, y_test)
    save_model(model)