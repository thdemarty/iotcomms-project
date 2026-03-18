import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

df_ = pd.pivot_table(df, values='rssi', index=['tx_node_id', 'timestamp'], columns=['env_id'])

node0col = ['#053c70', '#064b8c', '#386fa3', '#6a93ba', '#9bb7d1']
node1col = ['#05701e', '#068c25', '#38a351', '#6aba7c', '#9bd1a8']
node2col = ['#6d7005', '#888c06', '#a0a338', '#b8ba6a', '#cfd19b']
node3col = ['#701f05', '#8c2706', '#a35238', '#ba7d6a', '#d1a99b']
node4col = ['#6b0570', '#86068c', '#9e38a3', '#b66aba', '#cf9bd1']
nodecols = [node0col, node1col, node2col, node3col, node4col]
nodemarkers = ["s", "o", "^", "P", "*"]

for node in [0]:
    for env in [4]:
        # array = np.diff(np.asarray(df_.loc[node,env].dropna().index))
        # u, ind = np.unique(array, return_index=True)
        # plt.scatter(u, ind)
        # print(array[array<=30].shape)
        plt.scatter(df_.loc[node,env].dropna().index, df_.loc[node,env].dropna().values, s=2, c=nodecols[env][3], marker=nodemarkers[node])
        print(node, env, df_.loc[node,env].dropna().index[:3], df_.loc[node,env].dropna().index[-3:])


plt.show()