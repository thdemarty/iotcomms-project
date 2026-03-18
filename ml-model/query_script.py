import pandas as pd
import numpy as np

def dataframe_query(file_path, origin_ids, env_ids):

    df = pd.read_csv(file_path)

    filtered_df = df[(df["sender_id"].isin(origin_ids)) & (df["env_id"].isin(env_ids))]

    filtered_df = filtered_df[["receiver_id", "timestamp", "rssi"]]

    filtered_df = filtered_df.rename(columns={"timestamp": "timestep"})

    # timestep needs to be calculated per reciever per node!
    for receiver in [0,1,2,3,4]:
        filtered_df["timestep"][filtered_df["receiver_id"] == receiver] = np.append(0, np.diff(filtered_df["timestep"][filtered_df["receiver_id"] == receiver].values))
    filtered_df["timestep"][filtered_df["timestep"] < 0] = 0
    filtered_df["timestep"][filtered_df["timestep"] > 200] = 0

    return filtered_df

if __name__ == "__main__":
    path = "data/dataset.csv"
    df = dataframe_query(path, [0,1,2,3,4], [0,1,2,3,4])

    print(df)