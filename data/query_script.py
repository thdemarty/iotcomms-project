import pandas as pd

def dataframe_query(file_path, origin_ids, env_ids):

    df = pd.read_csv(file_path)

    filtered_df = df[(df["tx_node_id"].isin(origin_ids)) & (df["env_id"].isin(env_ids))]

    filtered_df = filtered_df[["node_id", "timestamp", "rssi"]]

    filtered_df = filtered_df.rename(columns={"timestamp": "timestep"})

    min_time = filtered_df["timestep"].min()
    filtered_df["timestep"] = filtered_df["timestep"] - min_time

    return filtered_df