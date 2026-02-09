import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Import existing preprocessing function
from prepare_data import preprocessing

def create_dataset():
    """
    Simulates loading data for multiple environments (Classes)
    In a real scenario, you would loop through your generated CSV files.
    """
    X = [] # Features
    y = [] # Labels (e.g., 0 for Normal, 1 for Interference)

    files = [
        {"path": "../data/test_ts_node1_env1.csv", "label": 0},
        {"path": "../data/test_ts_node1_env2.csv", "label": 1}
    ]

    print("Loading and preprocessing data...")
    
    for file_info in files:
        if not os.path.exists(file_info["path"]):
            print(f"Warning: {file_info['path']} not found. Please run create_test_data.py first.")
            continue
            
        rssi_norm, lqi_norm = preprocessing(file_info["path"])
        
        features = np.column_stack((rssi_norm, lqi_norm))
        
        labels = np.full(len(features), file_info["label"])
        
        if len(X) == 0:
            X = features
            y = labels
        else:
            X = np.vstack((X, features))
            y = np.hstack((y, labels))

    return X, y

def train():
    # 1. Prepare Data
    X, y = create_dataset()
    
    if len(X) == 0:
        print("No data found. Exiting.")
        return

    # 2. Split into Training and Testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Initialize the Model
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # 4. Train the Model
    print(f"Training model on {len(X_train)} samples...")
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Env 1", "Env 2"]))

    # 6. Save the model for later use
    joblib.dump(clf, 'link_quality_model.pkl')
    print("Model saved to 'link_quality_model.pkl'")

if __name__ == "__main__":
    train()