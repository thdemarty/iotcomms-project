import numpy as np
import pandas as pd
import re
from numpy.typing import NDArray

def preprocessing(path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Preprocessing the raw data.
    -----
    Time-series data is first differentiated (x(t+1) - x(t)) and then normalised (from 0.0 to 1.0).\n
    - path: path of csv recoding containing RSSI and LQI data
    """
    # read in file
    df = pd.read_csv(path)

    # differentiate RSSI and LQI
    rssi = np.diff(df["RSSI"])
    lqi = np.diff(df["LQI"])

    # normalise
    rssi = (rssi - np.min(rssi)) / (np.max(rssi) - np.min(rssi))
    lqi = (lqi - np.min(lqi)) / (np.max(lqi) - np.min(lqi))

    return rssi, lqi

def segmentation(rssi: NDArray[np.float64], lqi: NDArray[np.float64], frame_size: int, overlap: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Segmentation of the time-series data.
    -----
    - rssi: rssi time-series
    - lqi: lqi time-series belonging to rssi
    - frame_size: how many datapoints per frame in s,
    - overlap: the portion shared between two adjacent frames (0.0...1.0)

    Since it is assumed, that the data really comes in at 10 Hz, frame calculation is simplified!
    """
    # length of time-series (should be equal for both) in ticks (10 ticks/second)
    ts_length = rssi.shape[0]
    
    # frame length in ticks
    frame_length = frame_size * 10

    # create frames
    total_frames = int((ts_length - frame_length) / (frame_length * overlap)) + 1
    rssi_frames = np.zeros((total_frames, frame_length))
    lqi_frames = np.zeros((total_frames, frame_length))
    for i in range(total_frames):
        lower = int(i * frame_length * overlap)
        upper = int(lower + frame_length)
        rssi_frames[i] = rssi[lower:upper]
        lqi_frames[i] = lqi[lower:upper]

    return rssi_frames, lqi_frames

if __name__ == "__main__":
    node_id = 1
    env = 1
    file_name = f"test_ts_node{node_id}_env{env}.csv"
    file_pattern = r"/ml-model/prepare_data.py$"
    path = re.sub(file_pattern, "/data/" + file_name, __file__)

    rssi, lqi = preprocessing(path)

    for array in (rssi, lqi):
        print(array, np.min(array), np.max(array))

    rssi_frames, lqi_frames = segmentation(rssi, lqi, 10, 0.5)

    for array in (rssi_frames, lqi_frames):
        print(array, array.shape)