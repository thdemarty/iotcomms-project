import re
import numpy as np
import pandas as pd
from perlin_noise import PerlinNoise, Interp

strategie = 1
node_id = 1
env = 1
time_start = np.random.randint(1000, 10000000)
total_nodes = 5
duration = 30 * 60 * 10 * (total_nodes - 1) # 30 minutes * 60 seconds/minute * 10 measurements/second * 4 receivers = duration in measurements

file_pattern = r"/ml-model/create_test_data.py$"
file_name = f"test_ts_node{node_id}_env{env}.csv"

# create pandas DataFrame (for NodeID,Timestamp,RSSI,LQI)
df = pd.DataFrame()

# create NodeID
df["NodeID"] = np.ones((duration)) * node_id

# create Timestamp
ts = [time_start]
for _ in range(duration - 1):
    time_start += np.random.randint(1,10)
    ts.append(time_start)
df["Timestamp"] = np.asarray(ts)

# create RSSI
noise = PerlinNoise(np.random.randint(0,255), amplitude=0.5, frequency=10, octaves=5, interp=Interp.COSINE, use_fade=False)
df["RSSI"] = np.array([noise.get(i) for i in df["Timestamp"]])

# create LQI
noise = PerlinNoise(np.random.randint(0,255), amplitude=1, frequency=10, octaves=5, interp=Interp.COSINE, use_fade=False)
df["LQI"] = np.array([noise.get(i) for i in df["Timestamp"]])

# save to file
df.to_csv(re.sub(file_pattern, "/data/" + file_name, __file__))