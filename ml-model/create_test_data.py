import re
import numpy as np
import pandas as pd
from perlin_noise import PerlinNoise, Interp

strategie = 1
time_start = 0
total_nodes = 5
environments = ["forest", "garden", "river", "lake", "bridges"]
duration = 30 * 60 * 10 * total_nodes * len(environments) # 30 minutes * 60 seconds/minute * 10 measurements/second * 5 senders * 5 environments = duration in measurements

file_pattern = r"/ml-model/create_test_data.py$"
file_name = f"test_ts.csv"

# create pandas DataFrame (for NodeID,Timestamp,RSSI,LQI)
df = pd.DataFrame()

# create Timestamp
ts = [time_start]
for _ in range(duration - 1):
    time_start += np.random.randint(1,10)
    ts.append(time_start)
df["Timestamp"] = np.asarray(ts)

# create NodeIDs
df["NodeID"] = np.concatenate([np.ones(int(duration / total_nodes)) * node_id for node_id in range(total_nodes)])

# create Envs
df["Environment"] = np.hstack((np.concatenate([np.ones(int(duration / (total_nodes * len(environments)))) * id for id in range(len(environments))]), ) * total_nodes)

rssi = []
lqi = []
for node_id in range(total_nodes):
    for env_id in range(len(environments)):
        # create RSSI
        noise = PerlinNoise(np.random.randint(0,255), amplitude=0.5, frequency=10, octaves=5, interp=Interp.COSINE, use_fade=False)
        rssi.append(np.array([noise.get(i) for i in df["Timestamp"][int(node_id*duration/total_nodes) + int(env_id*duration/(total_nodes*len(environments))):int(node_id*duration/total_nodes) + int((env_id+1)*duration/(total_nodes*len(environments)))]]))

        # create LQI
        noise = PerlinNoise(np.random.randint(0,255), amplitude=1, frequency=10, octaves=5, interp=Interp.COSINE, use_fade=False)
        lqi.append(np.array([noise.get(i) for i in df["Timestamp"][int(node_id*duration/total_nodes) + int(env_id*duration/(total_nodes*len(environments))):int(node_id*duration/total_nodes) + int((env_id+1)*duration/(total_nodes*len(environments)))]]))
df["RSSI"] = np.concatenate(rssi)
df["LQI"] = np.concatenate(lqi)

# save to file
df.to_csv(re.sub(file_pattern, "/data/" + file_name, __file__), index=False)