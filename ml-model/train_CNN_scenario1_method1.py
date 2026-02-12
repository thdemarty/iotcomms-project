from CNN_library import *
from dotenv import load_dotenv
import os
import re

"""
Script for training the CNN.
Scenario: 1
-> identify environments
Method: 1
-> mix and split all nodes 0.75 train 0.25 test
"""

### Load environment variables
file_pattern = r"[^/]*\.py$"
file_name = re.findall(file_pattern, __file__)[0].split(".py")[0]
cells = re.findall(r"[^_ .-]+", file_name)
save_path = re.sub(file_pattern, "saves/" + "_".join([cells[cell] for cell in [1,2,3]]) + ".pth", __file__)

load_dotenv(dotenv_path=re.sub(file_pattern, ".env", __file__))
TEST_SIZE = float(os.getenv('TEST_SIZE'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
DEVICE = os.getenv('DEVICE')
NUM_DEVICES = int(os.getenv('NUM_DEVICES'))
NUM_ENVS = int(os.getenv('NUM_ENVS'))
EPOCHS = int(os.getenv('EPOCHS'))
FRAME_SIZE = int(os.getenv('FRAME_SIZE'))
OVERLAP = float(os.getenv('OVERLAP'))

### Train / test split
dataset = BLEDataset("test_ts.csv", Method.MIX4TRAIN, Mode.ENV, window_size=FRAME_SIZE, overlap=OVERLAP)

indices = np.arange(len(dataset))
print(len(dataset))
train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

train_subset = torch.utils.data.Subset(dataset, train_idx)
test_subset = torch.utils.data.Subset(dataset, test_idx)

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
plot_confusion(cm_device, labels=dataset.device_encoder.classes_, title="Device Classification Confusion Matrix")
cm_env = confusion_matrix(env_true, env_pred)
plot_confusion(cm_env, labels=dataset.env_encoder.classes_, title="Environment Classification Confusion Matrix")

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

torch.save(model.state_dict(), save_path)