import pandas as pd

environments = {0:"garden",1:"forest",2:"lake",3:"river",4:"bridge"}

def dataframe_query(file_path, origin_ids, env_ids):

    df = pd.read_csv(file_path)

    filtered_df = df.loc[(df["sender_id"].isin(origin_ids)) & (df["env_id"].isin(env_ids))]

    filtered_df = filtered_df.loc[:,["receiver_id", "timestamp", "rssi"]]

    filtered_df = filtered_df.rename(columns={"timestamp": "timestep"})

    # timestep needs to be calculated per reciever per node!
    # normalize timestep
    filtered_df["timestep"] = filtered_df.groupby("receiver_id")["timestep"].diff().fillna(0)

    return filtered_df

# create node directory csv's (which are environments apparently)
for env in [0,1,2,3,4]:
    df = dataframe_query("data/dataset.csv", [0,1,2,3,4], [env])
    df.to_csv(f"data/node/{environments[env]}.csv", index=False)

# create env csv's
for node in [0,1,2,3,4]:
    df = dataframe_query("data/dataset.csv", [node], [0,1,2,3,4])
    df.to_csv(f"data/environment/node{node}.csv", index=False)
