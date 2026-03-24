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
TODO
### Models & Parameters
### Results
