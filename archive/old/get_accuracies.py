#!/usr/bin/env python3
import numpy as np
import pandas as pd

def nice_name(row):
    if row["model"] == "RandomForestClassifier":
        name = "RF"
    elif row["model"] == "ExtraTreesClassifier":
        name = "ET"
    elif row["model"] == "AdaBoostClassifier":
        name = "AB"

    if row["model"] == "SGDEnsemble":
        name = "PSGD"
        if row["forest_options"]["model"] == "RandomForestClassifier":
            base_name = "RF"
        elif row["forest_options"]["model"] == "ExtraTreesClassifier":
            base_name = "ET"
        elif row["forest_options"]["model"] == "AdaBoostClassifier":
            base_name = "AB"
        name += ", " + base_name + ", \lambda = " + str(row["optimizer"]["lambda"])
    # else:
        #name += #", " #+ "str(row["n_estimators"])

    return name

#path = "./magic/results/01-06-2020-10:42:28/results.jsonl"
path = "./magic/results/02-06-2020-22:21:06/results.jsonl"

df = pd.read_json(path, lines=True)
df["nice_name"] = df.apply(nice_name,axis=1)
# print(df)
#print(df["scores"].values)

train_accuracies = []
test_accuracies = []
nonzero = []
names = []

for scores,nice_name in zip(df["scores"].values, df["nice_name"]):
    train_accuracies.append(scores["mean_accuracy_train"])
    test_accuracies.append(scores["mean_accuracy_test"])
    nonzero.append(scores["mean_nonzero_weights_test"])
    names.append(nice_name)
    # accuracies.append((nice_name, scores["mean_accuracy_test"], scores["mean_nonzero_weights_test"]))
    # print((nice_name, np.round(scores["mean_accuracy_test"],decimals=3), scores["mean_nonzero_weights_test"]))

dff = pd.DataFrame(list(zip(names,train_accuracies,test_accuracies,nonzero)), columns=["method", "train accuracy", "test accuracy", "K"]).astype({'K': 'int32'})
dff = dff.round(3)
pd.set_option('display.width', 120)
print(dff.to_markdown())
# print(accuracies)