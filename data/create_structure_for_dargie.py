import pandas as pd

import sys
sys.path.insert(1, 'ml-model')
from query_script import dataframe_query

environments = {0:"garden",1:"forest",2:"lake",3:"river",4:"bridge"}

# create node directory csv's (which are environments apparently)
for env in [0,1,2,3,4]:
    df = dataframe_query("data/dataset.csv", [0,1,2,3,4], [env])
    df.to_csv(f"data/node/{environments[env]}.csv")

# create env csv's
for node in [0,1,2,3,4]:
    df = dataframe_query("data/dataset.csv", [node], [0,1,2,3,4])
    df.to_csv(f"data/environment/node{node}.csv")
