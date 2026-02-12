from CNN import *

file_pattern = r"[^/]*\.py$"
file_name = "CNN_scenario1_method1.pth"
path = re.sub(file_pattern, "saves/" + file_name, __file__)

model = BLECNN(num_devices=5, num_envs=5)
model.load_state_dict(torch.load(path, weights_only=True))
model.eval()


dataset = BLEDataset("test_ts.csv", Method.MIX4TRAIN, Mode.ENV)

test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

dev_true, dev_pred, env_true, env_pred = model.collect_predictions(test_loader, torch.device("cpu"))

cm_device = confusion_matrix(dev_true, dev_pred)
plot_confusion(cm_device, labels=dataset.device_encoder.classes_, title="Device Classification Confusion Matrix")
cm_env = confusion_matrix(env_true, env_pred)
plot_confusion(cm_env, labels=dataset.env_encoder.classes_, title="Environment Classification Confusion Matrix")
