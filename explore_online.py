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
    "ExtremelyFastDecisionTreeClassifier":"HTT",
    "HoeffdingTreeClassifier":"HT",
    "AdaptiveRandomForestClassifier":"ARF",
    "AdaBoostClassifier":"AB",
    "BaggingClassifier":"Bag",
    "SEModel":"SE",
    "JaxModel":"SDT",
    "TorchModel":"SDT",
    "WindowedTree":"SE",
    "moa.classifiers.meta.AdaptiveRandomForest":"ARF",
    "moa.classifiers.trees.HoeffdingTree":"HT",
    "moa.classifiers.trees.EFDT":"HTT",
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
    # elif (row["model"] == "SEModel"):
    #     model_name = "{} b = {} l = {}".format(name_mapping[row["model"]], row.get("model_params.burnin_steps", None), row.get("model_params.loss", None)) # row.get("model_params.additional_tree_options.max_features", None), row.get("model_params.step_size", None)
    # elif (row["model"] == "WindowedTree"):
    #     tree_init_mode = "None"
    #     if row.get("model_params.additional_tree_options.tree_init_mode", None) is not None:
    #         tree_init_mode = row.get("model_params.additional_tree_options.tree_init_mode")
    #     model_name = "{} {}".format(name_mapping[row["model"]], tree_init_mode)
    # elif (row["model"] == "SEModel"):
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
    "elec" ,"gas-sensor" , "weather", "nomao",  "covtype", "airlines", "led_a","led_g", "rbf_f", "rbf_m", "agrawal_a", "agrawal_g"
]

max_kb = [None, 10 * 1024, 1024, 128]
print_individual = False

combined = []
for d in datasets:
    # Skip experiments which have not yet been performed
    dataset_path = os.path.join(base_path, d, "results")
    if not os.path.isdir(dataset_path):
        continue
    all_subdirs = [os.path.join(dataset_path,di) for di in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, di))]
    latest_folder = max(all_subdirs, key=os.path.getmtime)

    print("Reading {}".format(os.path.join(latest_folder, "results.jsonl")))
    dff = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
    dff["nice_name"] = dff.apply(nice_name, axis=1)
    dff["dataset"] = d
    dff = dff.round(decimals = 3)

    print("Found {} experiments for {} dataset".format(len(dff), d))
    traindfs = []
    mean_accuracy = []
    mean_kappa = []
    mean_kappaM = []
    mean_kappaT = []
    mean_kappaC = []
    mean_loss = []
    mean_nodes = []
    mean_time = []
    mean_trees = []
    mean_memory = []

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
                "num_nodes":metrics["num_nodes"],
                "num_trees":metrics["num_trees"],
                "num_bytes":metrics["num_bytes"],
                "loss":metrics["loss"],
                "item_cnt":metrics["item_cnt"],
                "time":metrics["time_sum"],
                "loss_average":metrics["loss_sum"] / metrics["item_cnt"],
                "num_nodes_average":metrics["num_nodes_sum"] / metrics["item_cnt"],
                "num_trees_average":metrics["num_trees_sum"] / metrics["item_cnt"],
                "num_bytes_average":metrics["num_bytes_sum"] / metrics["item_cnt"],
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
            mean_nodes.append(tmp["num_nodes_average"][-1])
            mean_trees.append(tmp["num_trees_average"][-1])
            mean_memory.append(tmp["num_bytes_average"][-1])
            #mean_params.append(tmp["num_parameters_average"][-1])
        except Exception as e:
            print(e)
            #traindfs.append(pd.DataFrame())
            mean_accuracy.append(0)
            mean_nodes.append(0)
            mean_memory.append(0)
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
    dff["mean_nodes"] = mean_nodes
    dff["mean_trees"] = mean_trees
    dff["mean_memory"] = mean_memory 
    dff["mean_memory"] /= (1024.0) # Bytes to KB
    dff["train_details"] = traindfs

    # tabledf = dff[["dataset","nice_name", "mean_accuracy", "mean_kappa", "mean_kappaM", "mean_kappaT","mean_kappaC","scores.mean_fit_time","mean_nodes","mean_memory"]]
    combined.append(dff)

    # if print_individual:
    #     for kb in max_kb:
    #         if kb is not None:
    #             filteredDF = tabledf.drop(tabledf[tabledf.mean_memory > kb].index)
    #         else:
    #             filteredDF = tabledf
    #         filteredDF = filteredDF.sort_values(by=['mean_accuracy'], ascending = False)
    #         filteredDF = filteredDF.groupby('nice_name').head(1)#.reset_index(level=1, drop=True)
    #         print("{} experiments on {} after filtering for {} KB".format(len(filteredDF), d, kb))
    #         display(HTML(filteredDF.to_html()))
    #         print("")
print("FILES READ")
df = pd.concat(combined)

# %%

def highlight(s):
    '''
    Nice styling of inline tables. This highlights the best accuracy.
    This helps a lot when reviewing the results. Probably has only effect if Jupyter / VSCode is used.
    '''
    accs = []
    for i in range(0, len(s)):
        accs.append(s[i])

    max_acc = np.nanmax(accs)

    style = []
    for acc in accs:
        if acc == max_acc:
            style.append('background-color: blue; text-align: left')
        else:
            style.append('')
        
    return style

# for kb in max_kb:
#     if kb is not None:
#         filteredDF = df.copy()
#         filteredDF.loc[filteredDF.mean_memory > kb, 'mean_accuracy'] = 0
#         #filteredDF = df.drop(df[df.mean_memory > kb].index)
#     else:
#         filteredDF = df
    
#     filteredDF = filteredDF.sort_values(by=['mean_accuracy'], ascending = False)
#     filteredDF = filteredDF.groupby(["nice_name", "dataset"]).head(1)

#     print("Filtered for {} KB".format(kb))
#     #display(HTML(filteredDF.to_html()))
#     pivotDF = pd.pivot_table(filteredDF, values = "mean_accuracy", index = ["dataset"], columns = ["nice_name"])
#     display( pivotDF.style.apply(highlight,axis=1) )
#     #display(HTML(pivotDF.to_html()))
#     print(pivotDF.to_latex())
#     print("")

dff = df.copy()

dff["time [s]"] = dff["scores.mean_fit_time"]
dff["nodes"] = dff["mean_nodes"]
dff["accuracy"] = 100.0*dff["mean_accuracy"]
dff["size [kb]"] = dff["mean_memory"]

#print(dff[["accuracy"]])
#asdf

for kb in max_kb:
    if kb is not None:
        filteredDF = dff.copy()
        filteredDF.loc[filteredDF.mean_memory > kb, ['accuracy', "size [kb]", "time [s]", "mean_nodes"]] = 0
        #filteredDF = df.drop(df[df.mean_memory > kb].index)
    else:
        filteredDF = dff
    
    filteredDF = filteredDF.sort_values(by=["dataset", 'accuracy'], ascending = False)
    filteredDF = filteredDF.groupby(["nice_name", "dataset"]).head(1)
    
    print("Filtered for {} KB".format(kb))
    #display(HTML(filteredDF[["dataset", "nice_name", "accuracy", "nodes", "time [s]", "size [kb]"]].to_html()))
    #asdf

    if kb is None:
        pivotDF = pd.pivot_table(filteredDF, values = ["size [kb]", "accuracy", "time [s]"], index = ["nice_name"], columns = ["dataset"]) 
        pivotDF["size [kb]"] = pivotDF["size [kb]"].astype(float).round(0).astype(int)
        pivotDF["accuracy"] = pivotDF["accuracy"].astype(float).round(3)
        pivotDF["time [s]"] = pivotDF["time [s]"].astype(float).round(0).astype(int)
        #pivotDF["mean_nodes"] = pivotDF["mean_nodes"].astype(float).round(0).astype(int)
        pivotDF = pivotDF.reorder_levels([1, 0], axis=1).sort_index(1)
        pivotDF = pivotDF.T
        display(pivotDF)
    else:
        pivotDF = pd.pivot_table(filteredDF, values = ["accuracy"], index = ["nice_name"], columns = ["dataset"]) 
        #pivotDF["mean_nodes"] = pivotDF["mean_nodes"].astype(float).round(0).astype(int)
        #pivotDF = pivotDF.reorder_levels([1, 0], axis=1).sort_index(1)
        pivotDF["accuracy"] = pivotDF["accuracy"].astype(float).round(3)
        pivotDF = pivotDF.T
        display(pivotDF)
    #display( pivotDF.style.apply(highlight,axis=1) )
    print(pivotDF.to_latex())
    print("")
    #pivotDF = pd.pivot_table(filteredDF, values = ["size [kb]", "accuracy"], index = ["dataset"], columns = ["nice_name"]) 

    #pivotDF = pivotDF.swaplevel(0,1,axis = 1)#.sort(axis = 1)
    #pivotDF = pivotDF.reorder_levels([1, 0], axis=1).sort_index(1)
    #display(pivotDF) #.swaplevel(0,2)


    # break
    # #display(HTML(pivotDF.to_html()))
    #break


# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle

import plotly.io as pio
pio.orca.config.use_xvfb = True

selected_datasets = [
    "led_a", "nomao"
    # ,"gas-sensor" ,  "nomao",  "covtype", "airlines", "led_a","led_g", "rbf_f", "rbf_m", "agrawal_a", "agrawal_g"
]

selected_methods = ["SE", "ARF", "SDT", "SRP", "HT", "HTT", "NB", "Bag", "SB"] 

# https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=9
paired = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
#['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
symbols = ["circle", "square", "diamond", "x", "hourglass", "star", "triangle-up", "hexagon", "triangle-down"]
colors = {}
markers = {}
for m,c,s in zip(dff["nice_name"].unique(), cycle(paired), cycle(symbols)):
    colors[m] = c
    markers[m] = s

for dataset in selected_datasets:
    print("PLOTTING {}".format(dataset))
    dff = df.copy()
    dff = dff.loc[dff["dataset"] == dataset]
    dff = dff.loc[dff['nice_name'].isin(selected_methods)]
    dff = dff.sort_values(by=['mean_accuracy'], ascending = False)
    dff = dff.groupby("nice_name").head(1)

    fig = make_subplots(rows=2, cols=1, subplot_titles=[dataset], horizontal_spacing = 0.03, vertical_spacing = 0.02)
    for index, row in dff.iterrows():
        m = row["nice_name"]
        tdf = row["train_details"]
        if len(tdf) > 10000:
            idx = np.linspace(0,len(tdf) - 1,10000,dtype=int)
            tdf = tdf.iloc[idx]

        #fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["loss_average"], mode="lines", name = m, marker=dict(color = colors[m])), row = 1, col = 1)
        
        running_acc = np.convolve(tdf["accuracy"], np.ones(32)/32 , mode='valid')
        fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = running_acc, mode="lines+markers", name = m, marker=dict(color = colors[m], symbol = markers[m], maxdisplayed=20, size=8)), row = 1, col = 1)

        #fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["accuracy_average"], mode="lines+markers", name = m, marker=dict(color = colors[m], symbol = markers[m], maxdisplayed=20, size=8)), row = 1, col = 1)
        #fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["accuracy_average"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 2, col = 1)
        fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["num_bytes_average"], mode="lines+markers", name = m, showlegend = False, marker=dict(color = colors[m], symbol = markers[m], maxdisplayed=20, size=8)), row = 2, col = 1)

    fig.update_xaxes(title_text="Number of items", row=2, col=1, title_font = {"size": 16})
    #fig.update_yaxes(title_text="Loss", row=1, col=1, title_font = {"size": 16})
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, title_font = {"size": 16})
    fig.update_yaxes(title_text="Memory [KB]", type="log",row=2, col=1, title_font = {"size": 16})
    
    fig.update_layout(
        template="simple_white",
        legend=dict(orientation="h",yanchor="bottom",y=-0.1,xanchor="left",x=0.22),
        margin={'l': 5, 'r': 20, 't': 20, 'b': 5},
        height=900, width=1100
    )
    #fig.show()
    fig.write_image("{}.pdf".format(dataset))
    print("PLOTTING {} DONE".format(dataset))
    # break

