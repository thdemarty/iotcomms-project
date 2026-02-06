import numpy as np
import pandas as pd
import re

def preprocessing(path):
    # read in file
    df = pd.read_csv(path)

    # differentiate RSSI and LQI
    rssi = np.diff(df["RSSI"])
    lqi = np.diff(df["LQI"])

    # normalise
    rssi = (rssi - np.min(rssi)) / (np.max(rssi) - np.min(rssi))
    lqi = (lqi - np.min(lqi)) / (np.max(lqi) - np.min(lqi))

    return rssi, lqi

if __name__ == "__main__":
    node_id = 1
    env = 1
    file_name = f"test_ts_node{node_id}_env{env}.csv"
    file_pattern = r"/ml-model/prepare_data.py$"
    path = re.sub(file_pattern, "/data/" + file_name, __file__)

    for array in preprocessing(path):
        print(array)
