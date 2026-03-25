# ResNet Model: Final Evaluation Results

This document presents the results for the **1D ResNet** architecture. Following the consensus of the project team, all RSSI values were processed as raw data (unnormalized) to preserve the physical signal characteristics.

---

## Scenario 1: Environment Prediction
**Objective:** Categorize the signal location into one of five environments: **Bridge, Forest, Garden, Lake, and River**.

### Method 1: Random Split
* **Overall Accuracy:** 51.71%
* **Overall F-Score:** 50.41%

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bridge** | 0.58 | 0.49 | 0.53 | 1515 |
| **Forest** | 0.38 | 0.54 | 0.44 | 1718 |
| **Garden** | 0.46 | 0.39 | 0.42 | 1448 |
| **Lake** | 0.36 | 0.21 | 0.26 | 1386 |
| **River** | 0.75 | 0.88 | **0.81** | 1680 |

**Visual Analysis:**
![Confusion Matrix S1 M1](saves/CM_S1_M1_RSSI.png)

### Method 2: Leave-One-Out (Generalization to New Nodes)
* **Overall Accuracy:** 46.16%
* **Overall F-Score:** 44.31%

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bridge** | 0.56 | 0.54 | 0.55 | 1148 |
| **Forest** | 0.39 | 0.21 | 0.28 | 1303 |
| **Garden** | 0.22 | 0.18 | 0.20 | 1110 |
| **Lake** | 0.29 | 0.43 | 0.35 | 1073 |
| **River** | 0.72 | 0.91 | **0.81** | 1274 |

**Visual Analysis:**
![Confusion Matrix S1 M2](saves/CM_S1_M2_RSSI.png)

---

## Scenario 2: Node Identification
**Objective:** Identify the specific transmitter (Sender ID) of the packet.

### Method 1: Random Split
* **Overall Accuracy:** 29.84%
* **Overall F-Score:** 28.92%

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Node 0** | 0.41 | 0.33 | 0.37 | 1631 |
| **Node 1** | 0.28 | 0.42 | 0.33 | 1640 |
| **Node 2** | 0.25 | 0.23 | 0.24 | 1555 |
| **Node 3** | 0.29 | 0.12 | 0.17 | 1437 |
| **Node 4** | 0.28 | 0.37 | 0.32 | 1484 |

**Visual Analysis:**
![Confusion Matrix S2 M1](saves/CM_S2_M1_RSSI.png)

---

## Technical Comparison: CNN vs. ResNet

Based on the CNN results provided by the team and the ResNet evaluation, the following conclusions have been reached:

1.  **Feature Extraction Performance:** Both architectures demonstrate optimal performance within the **River** environment. The ResNet achieved a significantly higher F1-score (**0.81**) compared to the CNN's **0.67**, indicating that residual blocks are more effective at identifying the distinct multi-path RSSI features characteristic of riparian environments.
2.  **Generalization Stability:** The CNN's F1-scores showed high volatility in Method 2 testing. In contrast, the ResNet demonstrated superior stability; for instance, the **River** F1-score remained constant at **0.81** when generalizing to a previously unseen node.
3.  **Temporal Feature Analysis:** Experimental results for the **ResNet + Timestep** configuration showed a marked decrease in performance (28–32% accuracy). This validates the decision to prioritize **RSSI Only** as the primary feature, as temporal data appears to introduce noise rather than actionable spatial information.
4.  **Project Recommendation:** It is recommended that the ResNet architecture be utilized for **Environment Mapping** due to its robust generalization capabilities. The standard CNN architecture remains a viable, lower-complexity alternative for localized **Node Verification** tasks where computational efficiency is prioritized.
