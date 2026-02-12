from CNN_library import *
from dotenv import load_dotenv
import os
import re

"""
Script for training the CNN.
Scenario: 1
-> identify environments
Method: 2
-> mix and train with 4 and split off one node for testing
"""

### Load environment variables
file_pattern = r"[^/]*\.py$"
file_name = re.findall(file_pattern, __file__)[0].split(".py")[0]
cells = re.findall(r"[^_ .-]+", file_name)
save_path = re.sub(file_pattern, "saves/" + "_".join([cells[cell] for cell in [1,2,3]]) + ".pth", __file__)

load_dotenv(dotenv_path=re.sub(file_pattern, ".env", __file__))
FILTER_ID = int(os.getenv('FILTER_ID'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
DEVICE = os.getenv('DEVICE')
NUM_DEVICES = int(os.getenv('NUM_DEVICES'))
NUM_ENVS = int(os.getenv('NUM_ENVS'))
EPOCHS = int(os.getenv('EPOCHS'))
FRAME_SIZE = int(os.getenv('FRAME_SIZE'))
OVERLAP = float(os.getenv('OVERLAP'))

### Train / test split
train_subset = BLEDataset("test_ts.csv", Method.MIX4TRAIN, Mode.ENV, filter_id=FILTER_ID, window_size=FRAME_SIZE, overlap=OVERLAP)
test_subset = BLEDataset("test_ts.csv", Method.MIX4TEST, Mode.ENV, filter_id=FILTER_ID, window_size=FRAME_SIZE, overlap=OVERLAP)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

### Training loop
device = torch.device(DEVICE)
model = BLECNN(num_devices=NUM_DEVICES, num_envs=NUM_ENVS, device=device).to(device)

### Run training
epochs = EPOCHS

model.start_training(train_loader, test_loader, epochs)

plot_training_curve(model.get_training_curve())

dev_true, dev_pred, env_true, env_pred = model.collect_predictions(test_loader)

cm_device = confusion_matrix(dev_true, dev_pred)
plot_confusion(cm_device, labels=np.arange(NUM_DEVICES), title="Device Classification Confusion Matrix")
cm_env = confusion_matrix(env_true, env_pred)
plot_confusion(cm_env, labels=np.arange(NUM_ENVS), title="Environment Classification Confusion Matrix")

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

torch.save(model.state_dict(), save_path)