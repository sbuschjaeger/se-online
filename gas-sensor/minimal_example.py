#!/usr/bin/env python3

import sys
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import river

print("Loading data")
dfs = []
for i in range(1,11):
    dfs.append( pd.read_csv(os.path.join("Dataset", "batch{}.dat".format(i)), header=None, delimiter = " ") )
df = pd.concat(dfs, axis=0, ignore_index=True)

# map classes to {0,1,2,3,4,5}
Y = df[0].values.astype(np.int32) - 1
df = df.drop([0], axis=1)

normalize = True
if normalize:
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.values.astype(np.float64))
else:
    X = df.values.astype(np.float64)
print("labels: ", set(Y))
print("data: ", X.shape)

model = river.tree.HoeffdingTreeClassifier(
    grace_period = 50,
    split_confidence = 0.01,
    leaf_prediction = "nba"
)

accuracy = 0
cnt = 0

with tqdm(total=len(X), ncols=100) as pbar:
    for x,y in zip(X,Y):
        xdict = {}
        for j, xj in enumerate(x):
            xdict["att_" + str(j)] = xj

        pred = model.predict_one(xdict)
        model.learn_one(xdict,y)
        accuracy += (pred == y)
        cnt += 1
        pbar.update(1)
        desc = '[{}/{}] accuracy {:2.4f}'.format(cnt, len(X), 100.0 * accuracy / cnt )
        pbar.set_description(desc)
