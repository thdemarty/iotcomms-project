# IoT Communication – BLE Project Report
## Task 
Collect connection quality parameters for five interconnected nodes communicating using Bluetooth Low Energy in five different environments.

Train and test two machine learning models to identify a node or an environment solely from those connection quality parameters.

## Data Collection
The setup was implemented on Adafruit Feather nRF52840 Sense boards using RIOT OS and the included NimBLE Bluetooth stack.

Each node was configured to send BLE packets with maximum transmit power.

We collected data for roughly 30 minutes for each environment at a packet rate of about 10 packets per second.

Node IDs are compiled into the application binary that runs on the node, e.g.:
```
NODEID=2 NODE_COUNT=5 make all flash term
```

The data is collected in the form of the terminal output of each node which is redircted to a file on the connected host device.

### Communication Protocol
Before two nodes can communicate, they must establish a BLE connection.
The application then starts a separate thread for receiving incoming packets and periodically (roughly every 100 ms) broadcasts packets to all other nodes.

Should a connection be lost at any time, the application immediately tries to reestablish it. 
In the case that this mechanism was not able to recover the connection, we manually reset all boards and repositioned them until all were able to connect again.

### Collected Data
The receiving node logs the following information for each incoming packet:
- Local timestamp (system time of the receiving node)
- Sending node ID (which is encoded in the node address)
- RSSI (using the ble_gap_conn_rssi method)

Other data points are not readily available and were thus not included.
LQI, as proposed in our task, is not a feature of BLE.
Latency measurements would have required synchronizing device clocks, which was beyond the scope of this work.

### Network Structure
The goal is to gather as much data as possible from nodes at the maximum distance from each other.
Hence, we formed a fully connected mesh network.

Since it is impossible to place five nodes such that each individual pair of nodes is at its maximum distance, we positioned the nodes in a pentagon.
Our ability to adhere to this structure varied across the different environments.

To give an example of how the real conditions affected the distance between nodes:
For the river environment, the nodes were situated on two sides of a small river, with three nodes on one side and two on the other.
However, the width of the river was such that there were only a few meters of margin before connections failed.
As a result, nodes on the same side of the river were a lot closer together compared to nodes on opposing sides.

TODO: add precise distances per environment


## Machine Learning Analysis

Machine Learning models were trained to get correlations between sender or environment and the signals rssi.
Therefore 2 scenarios had to be analysed:

- Scenario 1: The goal is to uniquely identify the deployment environments. For this scenario, mix all the RSSI time series belonging to the same environments.
- Scenario 2: The goal is to uniquely identify the sensor nodes. For this scenario, mix all the RSSI time series belonging to the same sensor node.

Also the data had to be slpit into training and testing data.

> [!IMPORTANT]  
> Written for scenario 2. For scenario 1, exchange Node and Environment.

- Method 1: Mix all datapoints sent from Node X and do it for every Node. Use say 75% for training and the rest 25% for testing. 
- Method 2: Mix the data from 4 of the environments belonging to Node X and do it for every Node. Use this for training and the remaining fifth environment for testing.

### CNN

The CNN is a 1D convolutional neural network that consists of two convolutional blocks, each comprising a 1D convolution with kernel size 5 and padding 2, followed by ReLU activation and 1D max pooling with a stride of 2. The feature maps are then flattened and passed through a fully connected layer with ReLU activation and 30% dropout, followed by a final linear layer mapping to the output classes.

```
self.net = nn.Sequential(
    nn.Conv1d(1, 16, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool1d(2),

    nn.Conv1d(16, 32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool1d(2),

    nn.Flatten(),

    nn.Linear((input_size // 4) * 32, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes)
)
```

Additionaly giving the model timesteps (how much time passed since the last packet) in `train_cnn_timestep.py` proved to be ineffective and rather removed learning altogether. So it was not further pursued.

#### Train & Test

For training the data was prepared to be **differentiated and normalized** from 0 to 1.
This data was split into **frames of 100** consecutive datapoints with an **overlap of 50%**.
The **batch size** was set to 64 and then decreased to **32** in the pursuit of getting method 2 to work (which helped a little bit).
Training was prolonged from 20 to **40 epochs** with a minor effect on method 1 and no real effect on method 2, since there was almost no learning effect.
The **learning rate** was changed from 1e-3 to in the end **5e-4**.

Scenarios 1 and 2 were trained with methods 1 and 2 respectively.
Also for method 2, all ids were left out for testing once.

#### Results
Likewise, you r report should have confusion matrices as well as performance tables F-score, accuracy, etc.

The results for CNN can be found in its entirety -> [link to CNN results](ml-model/CNN_resulst.md)

What can be seen is, that with a classic split of 75%/25% there is a diagonal forming on the confusion matrix for both scenarios and the accuracy is far above 20% (20% means basically guessing, since 100%/5 classes=20%). So the characteristics of the rssi per node and environment are something that was learned and is therefore learnable and distinct. However the minor amount of learning for method 2 shows, that the network has to see every node and every environment at least a few times for classification. So the nodes behave differently in environments and vice versa. But some leaving out some of the id's performes better than ohers, leading to some of the diagonal in the confusion matrix forming and accuracy above 20% at around 30%. That shows that some level of abstraction can be made overall, but is not reliable.

### ResNet


### Comparison
