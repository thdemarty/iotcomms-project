import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import sys
sys.path.insert(1, 'ml-model')
from query_script import dataframe_query 
   
seaborn.set_theme(style = 'whitegrid')  

df = dataframe_query("data/dataset.csv", [0,1,2,3,4], [0,1,2,3,4])

# node_id,env_id,tx_node_id,timestamp,rssi

node0col = ['#053c70', '#064b8c', '#386fa3', '#6a93ba', '#9bb7d1']
node1col = ['#05701e', '#068c25', '#38a351', '#6aba7c', '#9bd1a8']
node2col = ['#6d7005', '#888c06', '#a0a338', '#b8ba6a', '#cfd19b']
node3col = ['#701f05', '#8c2706', '#a35238', '#ba7d6a', '#d1a99b']
node4col = ['#6b0570', '#86068c', '#9e38a3', '#b66aba', '#cf9bd1']
nodecols = [node0col, node1col, node2col, node3col, node4col]
nodemarkers = ["s", "o", "^", "P", "*"]

sample = df.sample(100_000, random_state=0)
seaborn.violinplot(data=sample, y="timestep", bw_adjust=1, gridsize=500, inner="quartile")

plt.show()