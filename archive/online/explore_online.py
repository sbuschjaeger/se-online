# %%
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize 
import os
import json 
from IPython.display import display, HTML

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return json_normalize(data)

name_mapping = {
    "SRP":"SRP",
    "ExtremelyFastDecisionTreeClassifier":"ET",
    "HoeffdingTreeClassifier":"HT",
    "AdaptiveRandomForestClassifier":"ARF",
    "AdaBoostClassifier":"AB",
    "BaggingClassifier":"Bag",
    "PyBiasedProxEnsemble":"PE",
    "JaxModel":"SDT"
}

def nice_name(row):
    if row["model"] == "RiverModel":
        model_name = name_mapping[row["model_params.river_model"]]
        if row["model_params.river_model"] in ["SRP","BaggingClassifier", "AdaBoostClassifier"]:
            model_name += " + " + name_mapping[row["model_params.river_params.model"]]
    else:
        model_name = name_mapping[row["model"]]
    # elif (row["model"] == "PyBiasedProxEnsemble"):
    #     model_name = "{} with d = {}, R1 = {}, λ1 = {}, R2 = {}, λ2 = {}, bs = {}, lr = {}, lt {}".format(row["model"], row.get("model_params.max_depth", None), row.get("model_params.ensemble_regularizer", "None"), row.get("model_params.l_ensemble_reg", "None"), row.get("model_params.tree_regularizer", "None"), row.get("model_params.l_tree_reg", "None"), row.get("model_params.batch_size", None), row.get("model_params.step_size", None), row.get("model_params.update_trees", None))
    # elif row["model"] == "JaxModel":
    #     model_name = "{} with T = {}, d = {}, with temp_scaling = {}".format(row["model"], row["model_params.n_trees"], row["model_params.max_depth"], row.get("model_params.temp_scaling", None) )
    # else:
    #     model_name = "{} with T = {}, d = {}, modes = {}/{}, stepsize = {}".format(row["model"], row["max_trees"], row["max_depth"], row["init_mode"],row["next_mode"], row["step_size"])
    
    return model_name

#dataset = "elec"
#dataset = "internet_ads"
#dataset = "nomao"
dataset = "gas-sensor"

#dataset = "covtype"
#dataset = "activity"
dataset = os.path.join(dataset, "results")
all_subdirs = [os.path.join(dataset,d) for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
#print(all_subdirs)
latest_folder = max(all_subdirs, key=os.path.getmtime)

print("Reading {}".format(os.path.join(latest_folder, "results.jsonl")))
df = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
df["nice_name"] = df.apply(nice_name, axis=1)
df = df.round(decimals = 3)

traindfs = []
mean_accuracy = []
mean_loss = []
mean_params = []
mean_time = []
for index, row in df.iterrows(): 
    experiment_path = row["out_path"]
    metrics = np.load(os.path.join(row["out_path"], "epoch_0.npy"), allow_pickle=True).item()
    traindf = pd.DataFrame(metrics)
    mean_accuracy.append(np.mean(metrics["accuracy"]))
    mean_params.append(np.mean(metrics["num_parameters"]))
    # mean_loss.append(np.mean(metrics["loss"]))
    # mean_time.append(np.mean(metrics["time"]))

df["mean_accuracy"] = mean_accuracy
df["mean_params"] = mean_params
# df["mean_loss"] = mean_loss
# df["mean_time"] = mean_time

tabledf = df[["nice_name", "mean_accuracy", "mean_params", "scores.mean_fit_time"]]
tabledf = tabledf.sort_values(by=['mean_accuracy'], ascending = False)
print("Processed {} experiments".format(len(tabledf)))
display(HTML(tabledf.to_html()))

# print("Runtimes")
# tabledf = tabledf.sort_values(by=['scores.mean_fit_time'], ascending = False)
# display(HTML(tabledf.to_html()))

# print("Experiments per group")
# tmp = tabledf.groupby(['nice_name']).size()
# print(tmp)

# idx = tabledf.groupby(['nice_name'])['mean_accuracy'].transform(max) == tabledf['mean_accuracy']
# shortdf = tabledf[idx]

# print("Best configuration per group")

# display(HTML(shortdf.to_html()))



# %%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


def get_pareto(df, columns):
    first = df[columns[0]].values
    second = df[columns[1]].values

    # Count number of items
    population_size = len(first)
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if (first[j] >= first[i]) and (second[j] < second[i]):
            #if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    
    return df.iloc[population_ids[pareto_front]]
    # # Return ids of scenarios on pareto front
    # return population_ids[pareto_front]


for name, group in df.groupby(["nice_name"]):
    pdf = get_pareto(group, ["mean_accuracy", "mean_params"])
    pdf = pdf[["nice_name", "mean_accuracy", "mean_params", "scores.mean_fit_time"]]
    print(pdf)
    pdf = pdf.sort_values(by=['mean_accuracy'], ascending = False)
    plt.plot(pdf["mean_params"].values, pdf["mean_accuracy"], linestyle='solid', label=name)

plt.legend(loc="lower right")

#%%

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=9
paired = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'] 
colors = {}
for m,c in zip(df["nice_name"].values, paired):
    colors[m] = c

fig = make_subplots(rows=3, cols=1, subplot_titles=["Covtype"], horizontal_spacing = 0.03, vertical_spacing = 0.02)

for tdf, m in zip(traindfs, df["nice_name"].values):
    fig = fig.add_trace(go.Scatter(x=tdf["total_item_cnt"], y = tdf["item_loss"], mode="lines", name = m, marker=dict(color = colors[m])), row = 1, col = 1)
    fig = fig.add_trace(go.Scatter(x=tdf["total_item_cnt"], y = tdf["item_accuracy"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 2, col = 1)
    fig = fig.add_trace(go.Scatter(x=tdf["total_item_cnt"], y = tdf["item_num_parameters"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 3, col = 1)

fig.update_xaxes(title_text="Number of items", row=3, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="Loss", row=1, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="Accuracy", row=2, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="Num of trainable parameters", row=3, col=1, title_font = {"size": 16})

fig.update_layout(
    template="simple_white",
    legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="left",x=0.15),
    margin={'l': 5, 'r': 20, 't': 20, 'b': 5},
    height=900, width=1100
)
fig.show()


#%%
sub_experiments = [os.path.join(experiment_path,d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]

if len(sub_experiments) == 0:
    sub_experiments = [os.path.join(experiment_path, "epoch_0.npy")]

accuracies = []
losses = []
num_nodes = []
times = []
total_item_cnt = None
for experiment in sub_experiments:
    print("Reading {}".format(experiment))
    tdf = read_jsonl(experiment)
    losses.append(tdf["item_loss"].values)
    accuracies.append(tdf["item_accuracy"].values)
    num_nodes.append(tdf["item_num_parameters"].values)
    times.append(tdf["item_time"].values)
    if total_item_cnt is None:
        total_item_cnt = tdf["total_item_cnt"]

    
d = {
    "total_item_cnt":total_item_cnt,
    "item_loss":np.mean(losses, axis=0),
    "item_accuracy":np.mean(accuracies, axis=0),
    "item_num_parameters":np.mean(num_nodes, axis=0),
    "item_time":np.mean(times,axis=0)
}
