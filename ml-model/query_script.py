import pandas as pd

def dataframe_query(file_path, origin_ids, env_ids):

    df = pd.read_csv(file_path)

    filtered_df = df.loc[(df["sender_id"].isin(origin_ids)) & (df["env_id"].isin(env_ids))]

    filtered_df = filtered_df.loc[:,["receiver_id", "timestamp", "rssi"]]

    filtered_df = filtered_df.rename(columns={"timestamp": "timestep"})

    # timestep needs to be calculated per reciever per node!
    # normalize timestep
    filtered_df["timestep"] = filtered_df.groupby("receiver_id")["timestep"].diff().fillna(0)
    # normalize rssi
    filtered_df["rssi"] = filtered_df.groupby("receiver_id")["rssi"].diff().fillna(0)
    filtered_df["rssi"] = filtered_df.groupby("receiver_id")["rssi"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    # cut of big jumps
    filtered_df = filtered_df.loc[(filtered_df["timestep"] >= 0) & (filtered_df["timestep"] < 250)]

    return filtered_df

if __name__ == "__main__":
    path = "data/dataset.csv"
    df = dataframe_query(path, [0,1,2,3,4], [0])

    print(df)