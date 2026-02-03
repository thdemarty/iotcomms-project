## µC-Code
### Establish Network
steal some code from the BLE exercise
issue: adapt to 5 nodes → BLE Broadcast?

set transmission power

### Collect Datapoints
- 5 env * 30 min/env * 60 s/min / 10 s/packet = 900 packets (per µC)
 → practically nothing in terms of data
- RSSI and LQI → need to find values in RIOT
- long-term storage
	- reruns are insanely time consuming
	- write data to persistent storage (should not be fragile, e.g., wiped on pressing reset)
	- have some easy (and tested) way of getting data from to µC

## Data Collection
Design expriments to find maximum radio range

Trial run in APB
- dry testing → everything should be implemented
- we do not want to fix issues outside in the cold

Final collection
- 5 envs * (30 min/env measurement + 15 min/env setup) = 3h 45 min
- location: Großer Garten
	- chosen due to relative proximity of all relevant environments

## ML Analysis
For implementation: pure python is easier to collaborate on compared to jupyter notebooks

Can be prepared in parallel using mock data

### Preprocessing

### Training/ Testing
Scenarios: Identify environments + Identify sensor nodes
Models: CNN model + ResNet model

Evalute for precission and accuracy

### Final report
- Dargie only cares about the data → should not put too much effort in our own write up
- Present dimensions in confusion matrix
