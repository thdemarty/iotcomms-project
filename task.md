# Timeline 
2026-02-10: 
- finish work on subtasks
- if everything is working: run dry test

between 2026-02-16 and 2026-02-20:
- final measurement in Großer Garten

# TODO
## µC-Code
### Establish Network [Thomas, Cedric]
steal some code from the BLE exercise
issue: adapt to 5 nodes → BLE Broadcast?

set transmission power

RSSI and LQI → need to find values in RIOT

write easily deployable version

print CSV line by line

### Collect Datapoints [Justus]
- 5 env * 30 min/env * 60 s/min * 10 packet/s = 90000 packets (per µC)
 → practically nothing in terms of data
- long-term storage
	- reruns are insanely time consuming
	- write data to persistent storage (should not be fragile, e.g., wiped on pressing reset)
	- have some easy (and tested) way of getting data from to µC

Try collecting printed data from µC and pipe to file using connected laptop

## Data Collection
Design expriments to find maximum radio range

Trial run in APB
- dry testing → everything should be implemented
- we do not want to fix issues outside in the cold

Final collection
- 5 envs * (30 min/env measurement + 15 min/env setup) = 3h 45 min
- location: Großer Garten + Elbe
	- chosen due to relative proximity of all relevant environments


## ML Analysis [Teoman, Robert]
For implementation: pure python is easier to collaborate on compared to jupyter notebooks

Can be prepared in parallel using mock data (should be roughly ready by the time we get the actual data)

### Preprocessing

### Training/ Testing
Scenarios: Identify environments + Identify sensor nodes
Models: CNN model + ResNet model

Evalute for precission and accuracy

### Final report
- Dargie only cares about the data → should not put too much effort in our own write up
- Present dimensions in confusion matrix
