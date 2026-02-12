from CNN_library import *

# Paths to trained models
file_pattern = r"[^/]*\.py$"
path = re.split(file_pattern, __file__)[0] + "saves/"
paths = [path + save_file for save_file in ["CNN_scenario1_method1.pth"]]

# Trust weights
device_weights = [1.0]
env_weights    = [0.0]

# Load data
val_dataset = BLEDataset("test_ts.csv", Method.VALIDATE)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create ensemble with automatic per-metric weighting
model = CNN(paths, device_weights=device_weights, env_weights=env_weights, device="cpu")

# Evaluate on test set
dev_acc, env_acc = model.evaluate(val_loader)
print(f"Device Acc: {dev_acc:.3f}, Env Acc: {env_acc:.3f}")

# Get predictions
dev_true, dev_pred, env_true, env_pred = model.collect_predictions(val_loader)

cm_device = confusion_matrix(dev_true, dev_pred)
plot_confusion(cm_device, labels=val_dataset.device_encoder.classes_, title="Fused Device Classification Confusion Matrix")
cm_env = confusion_matrix(env_true, env_pred)
plot_confusion(cm_env, labels=val_dataset.env_encoder.classes_, title="Fused Environment Classification Confusion Matrix")

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())