# %%
import numpy as np
import pandas as pd
# from pandas.io.json import json_normalize 
import os
import json 
from IPython.display import display, HTML
import gzip
import pickle

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return pd.json_normalize(data)

name_mapping = {
    "SRP":"SRP",
    "ExtremelyFastDecisionTreeClassifier":"ET",
    "HoeffdingTreeClassifier":"HT",
    "AdaptiveRandomForestClassifier":"ARF",
    "AdaBoostClassifier":"AB",
    "BaggingClassifier":"Bag",
    "PrimeModel":"Prime",
    "JaxModel":"SDT",
    "WindowedTree":"WT",
    "moa.classifiers.meta.AdaptiveRandomForest":"ARF",
    "moa.classifiers.trees.HoeffdingTree":"HT",
    "moa.classifiers.trees.EFDT":"ET",
    "moa.classifiers.meta.StreamingRandomPatches":"SRP",
    "moa.classifiers.meta.OzaBag":"Bag",
    "moa.classifiers.bayes.NaiveBayes":"NB",
    "moa.classifiers.meta.OnlineSmoothBoost":"SB"
}

def nice_name(row):
    if row["model"] == "RiverModel":
        model_name = name_mapping[row["model_params.river_model"]]
        if row["model_params.river_model"] in ["SRP","BaggingClassifier", "AdaBoostClassifier"]:
            model_name += " + " + name_mapping[row["model_params.river_params.model"]]
    elif (row["model"] == "PrimeModel"):
        tree_init_mode = "None"
        if row.get("model_params.additional_tree_options.tree_init_mode", None) is not None:
            tree_init_mode = row.get("model_params.additional_tree_options.tree_init_mode")
        model_name = "{} {}".format(name_mapping[row["model"]], tree_init_mode)
    elif (row["model"] == "WindowedTree"):
        tree_init_mode = "None"
        if row.get("model_params.splitter", None) is not None:
            tree_init_mode = row.get("model_params.splitter")
        model_name = "{} {}".format(name_mapping[row["model"]], tree_init_mode)
    # elif (row["model"] == "PrimeModel"):
    #     tree_init_mode = "None"
        
    #     if row.get("model_params.additional_tree_options.tree_init_mode", None) is not None:
    #         tree_init_mode = row.get("model_params.additional_tree_options.tree_init_mode")
        
    #     if row.get("model_params.additional_tree_options.splitter", None) is not None:
    #         tree_init_mode = row.get("model_params.additional_tree_options.splitter", None)

    #     model_name = "{} d = {} λ1 = {} bs = {} lr = {} lt = {} b = {} ti = {} l = {}".format(
    #         name_mapping[row["model"]],
    #         row.get("model_params.additional_tree_options.max_depth", "None"),
    #         row.get("model_params.l_ensemble_reg", "None"),
    #         row.get("model_params.batch_size", "None"), 
    #         row.get("model_params.step_size", "None"),
    #         row.get("model_params.update_leaves", "None"), 
    #         row.get("model_params.backend", "None"),
    #         tree_init_mode,
    #         row.get("model_params.loss", "None")
    #     )
    # elif (row["model"] == "WindowedTree"):
    #     model_name = "{} d = {} bs = {} ti = {}".format(
    #         name_mapping[row["model"]],
    #         row.get("model_params.max_depth", "None"),
    #         row.get("model_params.batch_size", "None"), 
    #         row.get("model_params.splitter", "None"),
    #     )
    elif row["model"] == "MoaModel":
        model_name = name_mapping[row["model_params.moa_model"]]
    else:
        model_name = name_mapping[row["model"]]
    # elif row["model"] == "JaxModel":
    #     model_name = "{} with T = {}, d = {}, with temp_scaling = {}".format(row["model"], row["model_params.n_trees"], row["model_params.max_depth"], row.get("model_params.temp_scaling", None) )
    # else:
    #     model_name = "{} with T = {}, d = {}, modes = {}/{}, stepsize = {}".format(row["model"], row["max_trees"], row["max_depth"], row["init_mode"],row["next_mode"], row["step_size"])
    
    return model_name

base_path = "/rdata/s01b_ls8_000/buschjae/"

datasets = [
    "elec", "gmsc", "gas-sensor", "nomao","weather", "spam",
    "airlines", "covtype", "agrawal_a", "agrawal_g", "led_a", "led_g", "rbf_f", "rbf_m", 
]

for d in datasets:
    dataset_path = os.path.join(base_path, d, "results")
    all_subdirs = [os.path.join(dataset_path,di) for di in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, di))]
    latest_folder = max(all_subdirs, key=os.path.getmtime)

    print("Reading {}".format(os.path.join(latest_folder, "results.jsonl")))
    dff = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
    dff["nice_name"] = dff.apply(nice_name, axis=1)
    dff = dff.round(decimals = 3)
    combined = []
    print_individual = False

    print("Found {} experiments for {} dataset".format(len(dff), d))
    traindfs = []
    mean_accuracy = []
    mean_kappa = []
    mean_kappaM = []
    mean_kappaT = []
    mean_kappaC = []
    mean_loss = []
    mean_params = []
    mean_time = []
    mean_trees = []

    for index, row in dff.iterrows(): 
        try:
            experiment_path = row["out_path"]
            # with gzip.open(os.path.join(row["out_path"], "training.npy.gz"), 'rb') as ifp:
            #     print(ifp)
            #     metrics = np.load(ifp, allow_pickle=True)
            # print("READING {}".format(os.path.join(row["out_path"], "training.npy.gz")))
            gzip_file = gzip.GzipFile(os.path.join(row["out_path"], "training.npy.gz"), "rb")
            metrics = pickle.load(gzip_file)
            # metrics = np.load(gzip_file, allow_pickle=True)

            tmp = {
                "accuracy":metrics["accuracy"],
                "kappa":metrics["kappa"],
                "kappaM":metrics["kappaM"],
                "kappaT":metrics["kappaT"],
                "kappaC":metrics["kappaC"],
                #"num_parameters":metrics["num_parameters"],
                #"num_trees":metrics["num_trees"],
                "loss":metrics["loss"],
                "item_cnt":metrics["item_cnt"],
                "time":metrics["time_sum"],
                "loss_average":metrics["loss_sum"] / metrics["item_cnt"],
                #"num_parameters_average":metrics["num_parameters_sum"] / metrics["item_cnt"],
                #"num_trees_average":metrics["num_trees_sum"] / metrics["item_cnt"],
                "accuracy_average":metrics["accuracy_sum"] / metrics["item_cnt"],
                "kappa_average":metrics["kappa_sum"] / metrics["item_cnt"],
                "kappaM_average":metrics["kappaM_sum"] / metrics["item_cnt"],
                "kappaT_average":metrics["kappaT_sum"] / metrics["item_cnt"],
                "kappaC_average":metrics["kappaC_sum"] / metrics["item_cnt"]
            }

            traindf = pd.DataFrame(tmp)
            traindfs.append(traindf)
            # Although we only samle up to Nmax entries we make sure hat we also sample the last entry via np.linspace which contains the sum over the entire stream for the specific metrics. This way, we make sure to obtain the correct average
            mean_kappa.append(tmp["kappa_average"][-1])
            mean_kappaM.append(tmp["kappaM_average"][-1])
            mean_kappaT.append(tmp["kappaT_average"][-1])
            mean_kappaC.append(tmp["kappaC_average"][-1])
            mean_accuracy.append(tmp["accuracy_average"][-1])
            #mean_params.append(tmp["num_parameters_average"][-1])
            #mean_trees.append(tmp["num_trees_average"][-1])
        except Exception as e:
            print(e)
            #traindfs.append(pd.DataFrame())
            mean_accuracy.append(0)
            mean_params.append(0)
            mean_trees.append(0)
            mean_kappa.append(0)
            mean_kappaM.append(0)
            mean_kappaT.append(0)
            mean_kappaC.append(0)
        
    #print("Preparing tables")
    dff["mean_accuracy"] = mean_accuracy
    dff["mean_kappa"] = mean_kappa
    dff["mean_kappaM"] = mean_kappaM
    dff["mean_kappaT"] = mean_kappaT
    dff["mean_kappaC"] = mean_kappaC
    # df["mean_params"] = mean_params
    # df["mean_trees"] = mean_trees
    # df["train_details"] = traindfs

    tabledf = dff[["dataset","nice_name", "mean_accuracy", "mean_kappa", "mean_kappaM", "mean_kappaT","mean_kappaC","scores.mean_fit_time"]]
    tabledf = tabledf.sort_values(by=['mean_accuracy'], ascending = False)
    tabledf = tabledf.groupby('nice_name').head(1)#.reset_index(level=1, drop=True)
    #print("Processed {} experiments".format(len(tabledf)))
    combined.append(tabledf)
    if print_individual:
        display(HTML(tabledf.to_html()))
        print("")
    break

df = pd.concat(combined)
display(HTML(df.to_html()))

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle

idx = df.groupby(['nice_name'])['mean_accuracy'].transform(max) == df['mean_accuracy']
shortdf = df[idx].sort_values(by=['mean_accuracy'], ascending = False)

# https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=9
paired = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'] 
colors = {}
for m,c in zip(shortdf["nice_name"].values, cycle(paired)):
    colors[m] = c
fig = make_subplots(rows=5, cols=1, subplot_titles=[dataset], horizontal_spacing = 0.03, vertical_spacing = 0.02)

for index, dff in shortdf.head(n=5).iterrows():
    m = dff["nice_name"]
#for dff, m in zip(shortdf, shortdf["nice_name"].values):
    tdf = dff["train_details"]
    if len(tdf) > 10000:
        idx = np.linspace(0,len(tdf) - 1,10000,dtype=int)
        tdf = tdf.iloc[idx]

    fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["loss_average"], mode="lines", name = m, marker=dict(color = colors[m])), row = 1, col = 1)
    
    running_acc = np.convolve(tdf["accuracy"], np.ones(1024)/1024 , mode='valid')
    fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = running_acc, mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 2, col = 1)
    #fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["accuracy_average"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 2, col = 1)
    fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["num_parameters_average"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 3, col = 1)
    fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["num_trees_average"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 4, col = 1)
    fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["time"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 5, col = 1)

fig.update_xaxes(title_text="Number of items", row=5, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="Loss", row=1, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="Accuracy", row=2, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="No. trainable parameters", row=3, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="No. trees", row=4, col=1, title_font = {"size": 16})
fig.update_yaxes(title_text="Cumulative time [s]", row=5, col=1, title_font = {"size": 16})

fig.update_layout(
    template="simple_white",
    legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="left",x=0.15),
    margin={'l': 5, 'r': 20, 't': 20, 'b': 5},
    height=900, width=1100
)
fig.show()

#with pd.option_context('display.max_rows', None, 'display.max_columns', 250):  # more options can be specified also
#    print(tabledf)
# print("Runtimes")
# tabledf = tabledf.sort_values(by=['scores.mean_fit_time'], ascending = False)
# display(HTML(tabledf.to_html()))


# idx = tabledf.groupby(['nice_name'])['mean_accuracy'].transform(max) == tabledf['mean_accuracy']
# shortdf = tabledf[idx]
# shortdf = shortdf.sort_values(by=['mean_accuracy'], ascending = False)
# print("Best configuration per group")
# # display(HTML(shortdf.to_html()))
# #with pd.option_context('display.max_rows', None, 'display.max_columns', 250):  # more options can be specified also
# print(shortdf)



# # %%
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
# import numpy as np


# def get_pareto(df, columns):
#     first = df[columns[0]].values
#     second = df[columns[1]].values

#     # Count number of items
#     population_size = len(first)
#     # Create a NumPy index for scores on the pareto front (zero indexed)
#     population_ids = np.arange(population_size)
#     # Create a starting list of items on the Pareto front
#     # All items start off as being labelled as on the Parteo front
#     pareto_front = np.ones(population_size, dtype=bool)
#     # Loop through each item. This will then be compared with all other items
#     for i in range(population_size):
#         # Loop through all other items
#         for j in range(population_size):
#             # Check if our 'i' pint is dominated by out 'j' point
#             if (first[j] >= first[i]) and (second[j] < second[i]):
#             #if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
#                 # j dominates i. Label 'i' point as not on Pareto front
#                 pareto_front[i] = 0
#                 # Stop further comparisons with 'i' (no more comparisons needed)
#                 break
    
#     return df.iloc[population_ids[pareto_front]]
#     # # Return ids of scenarios on pareto front
#     # return population_ids[pareto_front]


# for name, group in df.groupby(["nice_name"]):
#     pdf = get_pareto(group, ["mean_accuracy", "mean_params"])
#     pdf = pdf[["nice_name", "mean_accuracy", "mean_params", "scores.mean_fit_time"]]
#     print(pdf)
#     pdf = pdf.sort_values(by=['mean_accuracy'], ascending = False)
#     plt.plot(pdf["mean_params"].values, pdf["mean_accuracy"], linestyle='solid', label=name)

# plt.legend(loc="lower right")




# #%%
# sub_experiments = [os.path.join(experiment_path,d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]

# if len(sub_experiments) == 0:
#     sub_experiments = [os.path.join(experiment_path, "epoch_0.npy")]

# accuracies = []
# losses = []
# num_nodes = []
# times = []
# total_item_cnt = None
# for experiment in sub_experiments:
#     print("Reading {}".format(experiment))
#     tdf = read_jsonl(experiment)
#     losses.append(tdf["item_loss"].values)
#     accuracies.append(tdf["item_accuracy"].values)
#     num_nodes.append(tdf["item_num_parameters"].values)
#     times.append(tdf["item_time"].values)
#     if total_item_cnt is None:
#         total_item_cnt = tdf["total_item_cnt"]

    
# d = {
#     "total_item_cnt":total_item_cnt,
#     "item_loss":np.mean(losses, axis=0),
#     "item_accuracy":np.mean(accuracies, axis=0),
#     "item_num_parameters":np.mean(num_nodes, axis=0),
#     "item_time":np.mean(times,axis=0)
# }

# %%

# import matplotlib as plt
from matplotlib.pyplot import plot

df = pd.read_csv(os.path.join("weather","weather.csv"))
Y = df["target"].values.astype(np.int32)
df = df.drop(["target"], axis=1)
is_nominal = (df.nunique() == 2).values
X = df.values.astype(np.float64)

Y = Y[3000:3128]
plot(range(0,len(Y)), Y)

# fig.update_xaxes(title_text="Number of items", row=5, col=1, title_font = {"size": 16})
# fig.update_yaxes(title_text="Loss", row=1, col=1, title_font = {"size": 16})
# fig.update_yaxes(title_text="Accuracy", row=2, col=1, title_font = {"size": 16})
# fig.update_yaxes(title_text="No. trainable parameters", row=3, col=1, title_font = {"size": 16})
# fig.update_yaxes(title_text="No. trees", row=4, col=1, title_font = {"size": 16})
# fig.update_yaxes(title_text="Cumulative time [s]", row=5, col=1, title_font = {"size": 16})

# fig.update_layout(
#     template="simple_white",
#     legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="left",x=0.15),
#     margin={'l': 5, 'r': 20, 't': 20, 'b': 5},
#     height=900, width=1100
# )

