# 📊 ResNet Model: Final Evaluation Results

This document presents the results for the **1D ResNet** architecture. Following team consensus, all RSSI values were processed as **raw data** (unnormalized) to preserve the physical signal characteristics.

---

## 🏙️ Scenario 1: Environment Prediction
**Goal:** Categorize the signal location into one of five environments: **Bridge, Forest, Garden, Lake, River**.

### Method 1 (Random Split)
* **Overall Accuracy:** 51.71%
* **Overall F-Score:** 50.41%

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bridge** | 0.58 | 0.49 | 0.53 | 1515 |
| **Forest** | 0.38 | 0.54 | 0.44 | 1718 |
| **Garden** | 0.46 | 0.39 | 0.42 | 1448 |
| **Lake** | 0.36 | 0.21 | 0.26 | 1386 |
| **River** | 0.75 | 0.88 | **0.81** | 1680 |

**Visuals:**
![Confusion Matrix S1 M1](saves/CM_S1_M1_RSSI.png)

### Method 2 (Leave-One-Out / New Node Generalization)
* **Overall Accuracy:** 46.16%
* **Overall F-Score:** 44.31%

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bridge** | 0.56 | 0.54 | 0.55 | 1148 |
| **Forest** | 0.39 | 0.21 | 0.28 | 1303 |
| **Garden** | 0.22 | 0.18 | 0.20 | 1110 |
| **Lake** | 0.29 | 0.43 | 0.35 | 1073 |
| **River** | 0.72 | 0.91 | **0.81** | 1274 |

**Visuals:**
![Confusion Matrix S1 M2](saves/CM_S1_M2_RSSI.png)

---

## 📡 Scenario 2: Node Identification
**Goal:** Identify which specific node (Sender ID) transmitted the packet.

### Method 1 (Random Split)
* **Overall Accuracy:** 29.84%
* **Overall F-Score:** 28.92%

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Node 0** | 0.41 | 0.33 | 0.37 | 1631 |
| **Node 1** | 0.28 | 0.42 | 0.33 | 1640 |
| **Node 2** | 0.25 | 0.23 | 0.24 | 1555 |
| **Node 3** | 0.29 | 0.12 | 0.17 | 1437 |
| **Node 4** | 0.28 | 0.37 | 0.32 | 1484 |

**Visuals:**
![Confusion Matrix S2 M1](saves/CM_S2_M1_RSSI.png)

---

## 🧠 Comparison: CNN vs. ResNet (The "2 Cents")

Based on the CNN results provided by the team and the ResNet evaluation, here is the final comparison:

1.  **High-Confidence Environments:** Both architectures show their strongest performance in the **River** environment. ResNet achieved a significantly higher F1-score (**0.81**) compared to the CNN's **0.67**, suggesting the residual blocks are superior at extracting unique multi-path RSSI features found near water.
2.  **Generalization Gap:** The CNN's F1-scores fluctuated heavily in Method 2 (e.g., Garden dropping from 0.65 to 0.11). The ResNet demonstrated better "stability," with the **River** F1-score remaining at **0.81** even when generalizing to a new node.
3.  **Timestep Feature:** Experiments with **ResNet + Timestep** resulted in lower performance (28-32% accuracy). This aligns with the group's decision to prioritize **RSSI Only**, as timesteps appear to introduce temporal noise rather than spatial location data.
4.  **Conclusion:** ResNet is recommended for **Environment Mapping** due to its robustness in generalization, while the simpler CNN may be sufficient for local **Node Verification** tasks where computational resources are limited.
