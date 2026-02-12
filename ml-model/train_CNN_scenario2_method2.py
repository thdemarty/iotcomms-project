from CNN import *
from dotenv import load_dotenv
import os
import re

"""
Script for training the CNN.
Scenario: 2
-> identify nodes
Method: 2
-> mix and train with 4 and split off one environment for testing
"""

### Load environment variables
file_pattern = r"[^/]*\.py$"
file_name = re.findall(file_pattern, __file__)[0].split(".py")[0]
cells = re.findall(r"[^_ .-]+", file_name)
save_path = re.sub(file_pattern, "saves/" + "_".join([cells[cell] for cell in [1,2,3]]) + ".pth", __file__)

load_dotenv(dotenv_path=re.sub(file_pattern, ".env", __file__))
FILTER_ID = int(os.getenv('FILTER_ID'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
DEVICE = os.getenv('DEVICE')
NUM_DEVICES = int(os.getenv('NUM_DEVICES'))
NUM_ENVS = int(os.getenv('NUM_ENVS'))
EPOCHS = int(os.getenv('EPOCHS'))
FRAME_SIZE = int(os.getenv('FRAME_SIZE'))
OVERLAP = float(os.getenv('OVERLAP'))

### Train / test split
train_subset = BLEDataset("test_ts.csv", Method.MIX4TRAIN, Mode.NODE, filter_id=FILTER_ID, window_size=FRAME_SIZE, overlap=OVERLAP)
test_subset = BLEDataset("test_ts.csv", Method.MIX4TEST, Mode.NODE, filter_id=FILTER_ID, window_size=FRAME_SIZE, overlap=OVERLAP)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

### Training loop (CPU only)
device = torch.device(DEVICE)
model = BLECNN(num_devices=NUM_DEVICES, num_envs=NUM_ENVS).to(device)

### Run training
epochs = EPOCHS

for epoch in range(epochs):
    train_loss = model.train_epoch(train_loader)
    dev_acc, env_acc = model.evaluate(test_loader)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss: {train_loss:.4f} | "
        f"Device Acc: {dev_acc:.3f} | "
        f"Env Acc: {env_acc:.3f}"
    )

dev_true, dev_pred, env_true, env_pred = model.collect_predictions(test_loader, device)

cm_device = confusion_matrix(dev_true, dev_pred)
plot_confusion(cm_device, labels=np.arange(NUM_DEVICES), title="Device Classification Confusion Matrix")
cm_env = confusion_matrix(env_true, env_pred)
plot_confusion(cm_env, labels=np.arange(NUM_ENVS), title="Environment Classification Confusion Matrix")