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

def nice_name(row):
    if row["model"] == "RiverModel":
        model_name = "{} with {} mode {}".format(row["model"], row["model_params.river_model"], row["river_model_params.leaf_prediction"])
    elif (row["model"] == "PyBiasedProxEnsemble"):
        model_name = "{} with max_depth {} and R1 = {}, λ1 = {}, R2 = {}, λ2 = {}".format(row["model"], row.get("model_params.max_depth", None), row.get("model_params.ensemble_regularizer", "None"), row.get("model_params.l_ensemble_reg", "None"), row.get("model_params.tree_regularizer", "None"), row.get("model_params.l_tree_reg", "None"))
    elif row["model"] == "JaxModel":
        model_name = "{} with T = {}, max_depth = {}, with temp_scaling = {}".format(row["model"], row["n_trees"], row["max_depth"], row["temp_scaling"])
    else:
        model_name = "{} with T = {}, max_depth = {}, modes = {}/{}, stepsize = {}".format(row["model"], row["max_trees"], row["max_depth"], row["init_mode"],row["next_mode"], row["step_size"])
    
    return model_name

dataset = "covtype"
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
for m in df["nice_name"].values:
    experiment_path = df.loc[ df["nice_name"] == m ]["out_path"].values[0]
    traindf = None
    
    sub_experiments = [os.path.join(experiment_path,d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]

    if len(sub_experiments) == 0:
        sub_experiments = [os.path.join(experiment_path, "training.jsonl")]

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
    traindf = pd.DataFrame(d)
    
    traindfs.append(traindf)
    mean_accuracy.append(np.mean(accuracies))
    mean_loss.append(np.mean(losses))
    mean_params.append(np.mean(num_nodes))
    mean_time.append(np.mean(times))

df["mean_accuracy"] = mean_accuracy
df["mean_loss"] = mean_loss
df["mean_params"] = mean_params
df["mean_time"] = mean_time

tabledf = df[["nice_name", "mean_accuracy", "mean_loss", "mean_params", "mean_time"]]
tabledf = tabledf.sort_values(by=['mean_accuracy'], ascending = False)
#display(tabledf)
display(HTML(tabledf.to_html()))

# %%
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