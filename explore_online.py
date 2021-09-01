# %%
import numpy as np
import pandas as pd
import os
import json 
from IPython.display import display, HTML
import gzip
import pickle

'''
This file offers four functionalities which are placed in three execution cells.
- The first cell (this cell) defines some helper function for reading the data and mapping the names of algorithms to something presentable for the paper (e.g. ExtremelyFastDecisionTreeClassifier -> HTT, HoeffdingTreeClassifier -> HT etc.). Also this cells reads the results on the selected datasets (see `datasets` below) and prepares the for furthere use. IMPORTANT: The reading porcess can take quite some time to finish, especially if there are many dataset. Even though we gzip the statistics of each model, they require a lot of memory on disk since a statistics for each data point is stored. Consequently, the loading also takes a good amount of time.
- The second cell perform the hyperparameter selection and filtering for different model sizes. It also produces the LaTex code for the tables which are in the paper
- The third cell plots the results on selected datasets
- The fourth cell computes the pareto fron and the area under the pareto fron. For plotting the CD diagram we used https://github.com/mirkobunse/CriticalDifferenceDiagrams.jl
'''

# Helper function to read the result file
def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return pd.json_normalize(data)

# Map the names of the classes to something presentable for the paper
name_mapping = {
    "SRP":"SRP",
    "ExtremelyFastDecisionTreeClassifier":"HTT",
    "HoeffdingTreeClassifier":"HT",
    "AdaptiveRandomForestClassifier":"ARF",
    "AdaBoostClassifier":"AB",
    "BaggingClassifier":"Bag",
    "SEModel":"SE",
    "PrimeModel":"SE",
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

# Map the names and configurations to something presentable for the paper
# This has mostly been used during the explorative phase of algorithm development. 
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

# Path to all results, e.g. the path which was set via the "--out_path" option. 
base_path = "/rdata/s01b_ls8_000/buschjae/"

# The datasets for which experiments have been performed on. 
datasets = [
    "gas-sensor", "elec", "weather","nomao","led_a","led_g", "rbf_f", "rbf_m", "agrawal_a", "agrawal_g", "covtype", "airlines"
]

# We can either search for the correct data-sets automatically (see below) or "hardcode" the correct path. 
pathes = {
    "agrawal_a":"results/04-06-2021-18:10:27",
    "agrawal_g":"results/05-06-2021-01:06:41",
    "airlines":"results/05-06-2021-21:33:24",
    "covtype":"results/05-06-2021-21:30:15",
    "elec":"results/04-06-2021-18:09:58",
    "gas-sensor":"results/04-06-2021-20:42:49",
    "led_a":"results/04-06-2021-18:10:13",
    "led_g":"results/05-06-2021-21:29:26",
    "nomao":"results/05-06-2021-01:29:41",
    "rbf_f":"results/04-06-2021-18:10:24",
    "rbf_m":"results/05-06-2021-21:29:27",
    "weather":"results/04-06-2021-23:18:09",
}

combined = []
for d in datasets:
    # Find the latest experiments automatically + Skip experiments which have not yet been performed
    # dataset_path = os.path.join(base_path, d, "results")
    # if not os.path.isdir(dataset_path):
    #     continue
    # all_subdirs = [os.path.join(dataset_path,di) for di in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, di))]
    # latest_folder = max(all_subdirs, key=os.path.getmtime)

    latest_folder = os.path.join(base_path, d, pathes[d])

    # The results.jsonl contains all experiments performed on this dataset and contains the path to each individual statistics file
    print("Reading {}".format(os.path.join(latest_folder, "results.jsonl")))
    dff = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
    dff["nice_name"] = dff.apply(nice_name, axis=1)
    dff["dataset"] = d
    dff = dff.round(decimals = 3)

    print("Found {} experiments for {} dataset".format(len(dff), d))
    # Prepare individual statistics
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

    # Extract all individual statistics over all experiments for this dataset
    for index, row in dff.iterrows(): 
        try:
            # Read the file for the current method
            experiment_path = row["out_path"]
            gzip_file = gzip.GzipFile(os.path.join(row["out_path"], "training.npy.gz"), "rb")
            metrics = pickle.load(gzip_file)
            
            # Extract the statistics
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
            mean_kappa.append(tmp["kappa_average"][-1])
            mean_kappaM.append(tmp["kappaM_average"][-1])
            mean_kappaT.append(tmp["kappaT_average"][-1])
            mean_kappaC.append(tmp["kappaC_average"][-1])
            mean_accuracy.append(tmp["accuracy_average"][-1])
            mean_nodes.append(tmp["num_nodes_average"][-1])
            mean_trees.append(tmp["num_trees_average"][-1])
            mean_memory.append(tmp["num_bytes_average"][-1])
        except Exception as e:
            print(e)
            mean_accuracy.append(0)
            mean_nodes.append(0)
            mean_memory.append(0)
            mean_trees.append(0)
            mean_kappa.append(0)
            mean_kappaM.append(0)
            mean_kappaT.append(0)
            mean_kappaC.append(0)
        
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

    # Add everything into one huge list
    combined.append(dff)

# Format list as pandas data frame
print("FILES READ")
df = pd.concat(combined)

# %%

"""
Output the test-then-train accuracy tables for different maximum model sizes. These tables only appear in the appendix. 
"""

# Since loading takes a long time we do not want to mess-up the dataframe. So we will first copy it
dff = df.copy()

# Rename some columns for nicer display
dff["time [s]"] = dff["scores.mean_fit_time"]
dff["nodes"] = dff["mean_nodes"]
dff["accuracy"] = 100.0*dff["mean_accuracy"]
dff["size [kb]"] = dff["mean_memory"]

# The filter options for the maximum model size in KB
max_kb = [None, 10 * 1024, 1024, 128]

# Print a table for each filtering step
for kb in max_kb:
    if kb is not None:
        # Perfrom the actual filtering
        filteredDF = dff.copy()
        filteredDF.loc[filteredDF.mean_memory > kb, ['accuracy', "size [kb]", "time [s]", "mean_nodes"]] = 0
    else:
        filteredDF = dff
    
    # Sort everything by the dataset and accuracy
    filteredDF = filteredDF.sort_values(by=["dataset", 'accuracy'], ascending = False)
    # Now group everything by the dataset and method (this preservers the original ordering) and pick the first group. This essentially selects the best (= highest accuracy) of each config on each dataset into a single frame 
    filteredDF = filteredDF.groupby(["nice_name", "dataset"]).head(1)
    
    print("Filtered for {} KB".format(kb))

    # Print pivot tables and round numbers to be a bit more readable
    if kb is None:
        pivotDF = pd.pivot_table(filteredDF, values = ["size [kb]", "accuracy", "time [s]"], index = ["nice_name"], columns = ["dataset"]) 
        pivotDF["size [kb]"] = pivotDF["size [kb]"].astype(float).round(0).astype(int)
        pivotDF["accuracy"] = pivotDF["accuracy"].astype(float).round(3)
        pivotDF["time [s]"] = pivotDF["time [s]"].astype(float).round(0).astype(int)
        pivotDF = pivotDF.reorder_levels([1, 0], axis=1).sort_index(1)
        pivotDF = pivotDF.T
        display(pivotDF)
    else:
        pivotDF = pd.pivot_table(filteredDF, values = ["accuracy"], index = ["nice_name"], columns = ["dataset"]) 
        pivotDF["accuracy"] = pivotDF["accuracy"].astype(float).round(3)
        pivotDF = pivotDF.T
        display(pivotDF)
    
    # Finally also output the tex code for the paper. Note that bold / italic entries have been added manually for the paper
    print(pivotDF.to_latex())
    print("")


# %%
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
from itertools import cycle

import plotly.io as pio
pio.orca.config.use_xvfb = True

"""
Output the test-then-train plots over the number of data points. For the paper we re-worked some plots manually in tikz. To do so, the CSV with the original data points are also stored.
"""


# Select the dataset for which we want to see individual graphs 
selected_datasets = [
    "gas-sensor", "led_a"
    # ,"gas-sensor" ,  "nomao",  "covtype", "airlines", "led_a","led_g", "rbf_f", "rbf_m", "agrawal_a", "agrawal_g"
]

# Select the methods for which we want to see individual graphs
selected_methods = ["ARF", "Bag", "HT", "HTT", "NB", "SB", "SDT", "SE", "SRP"] 

# Make sure that methods receive the same color and symbol across different plots
# Color coding has been taken from https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=9
# Marker coding has been taken from https://plotly.com/python/marker-style/
paired = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
symbols = ["circle", "square", "diamond", "x", "hourglass", "star", "triangle-up", "hexagon", "triangle-down"]
colors = {}
markers = {}
for m,c,s in zip(selected_methods, cycle(paired), cycle(symbols)):
    colors[m] = c
    markers[m] = s

# Plot each dataset
for dataset in selected_datasets:
    print("PLOTTING {}".format(dataset))
    dff = df.copy()

    # Select the appropriate dataset and methods. Then select the best (= highest accurcay) config for each method 
    dff = dff.loc[dff["dataset"] == dataset]
    dff = dff.loc[dff['nice_name'].isin(selected_methods)]
    dff = dff.sort_values(by=['mean_accuracy'], ascending = False)
    dff = dff.groupby("nice_name").head(1)

    # Prepare the figure for plotting
    fig = make_subplots(rows=2, cols=1, horizontal_spacing = 0.2, vertical_spacing = 0.1)
    
    # To make the plots look a little nicer in the paper we plotted them as pgf plots "by hand". To do so, we export the raw values
    all_accuracies = {}
    all_memories = {}
    for index, row in dff.iterrows():
        m = row["nice_name"]
        tdf = row["train_details"]

        # The accuracy for a single item is basically 0 or 1 which leads to a lot of jumps in the plots. This makes it difficult to distinguish anything at all. Hence we compute a "smoothing" here first
        tdf["running_accuracy"] = np.convolve(tdf["accuracy"], np.ones(16)/16 , mode='same')

        # Plotting takes a long time and looks a little strange if there are too many points. Thus we evenly select 10000 points 
        if len(tdf) > 10000:
            # Ignore the last 20 entries because they look weird due to the convolution
            idx = np.linspace(0,len(tdf) - 20,10000,dtype=int)
            tdf = tdf.iloc[idx]
        
        fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["running_accuracy"], mode="lines+markers", name = m, marker=dict(color = colors[m], symbol = markers[m], maxdisplayed=20, size=8)), row = 1, col = 1)
        fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["num_bytes_average"], mode="lines+markers", name = m, showlegend = False, marker=dict(color = colors[m], symbol = markers[m], maxdisplayed=20, size=8)), row = 2, col = 1)
        #fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["loss_average"], mode="lines", name = m, marker=dict(color = colors[m])), row = 1, col = 1)
        #fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["accuracy_average"], mode="lines+markers", name = m, marker=dict(color = colors[m], symbol = markers[m], maxdisplayed=20, size=8)), row = 1, col = 1)
        #fig = fig.add_trace(go.Scatter(x=tdf["item_cnt"], y = tdf["accuracy_average"], mode="lines", name = m, showlegend = False, marker=dict(color = colors[m])), row = 2, col = 1)

        # e export the raw values here
        if "item_cnt" not in all_accuracies:
            all_accuracies["item_cnt"] = tdf["item_cnt"]
            all_memories["item_cnt"] = tdf["item_cnt"]
        all_accuracies[m] = tdf["running_accuracy"]
        all_memories[m] = tdf["num_bytes_average"]

    # Store the raw values
    print("Writing CSV files")
    tmp = pd.DataFrame(all_accuracies)
    tmp.to_csv("accuracy_{}.csv".format(dataset), index=False)
    tmp = pd.DataFrame(all_memories)
    tmp.to_csv("memory_{}.csv".format(dataset), index=False)

    # Name the axes correctly and increase the font sizes for better readability
    fig.update_xaxes(title_text="Number of items", row=2, col=1, title_font = {"size": 20}, tickfont = {"size": 18})
    fig.update_xaxes(row=1, col=1, tickfont = {"size": 18})
    #fig.update_yaxes(title_text="Loss", row=1, col=1, title_font = {"size": 16})
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, title_font = {"size": 20}, tickfont = {"size": 18})
    fig.update_yaxes(title_text="Memory [KB]", type="log",row=2, col=1, title_font = {"size": 20}, tickfont = {"size": 18})
    
    # Choose a simple white layout + export the pdf files
    fig.update_layout(
        template="simple_white",
        legend=dict(orientation="h",yanchor="bottom",y=-0.18,xanchor="left",x=0.02,font = {"size": 20}),
        margin={'l': 5, 'r': 20, 't': 20, 'b': 5},
        height=700, width=900
    )
    fig.show()
    fig.write_image("{}.pdf".format(dataset))
    print("PLOTTING {} DONE".format(dataset))

# %%

import scipy
import matplotlib.pyplot as plt

"""
Compute the pareto front for each method on each dataset and store its (normalized) area under the pareto front for the table and plot presented in the paper. The table was directly copied into the paper. For plotting the CD diagram we used https://github.com/mirkobunse/CriticalDifferenceDiagrams.jl
"""

def get_pareto(df, columns):
    ''' Computes the pareto front of the given columns in the given dataframe. Returns results as a dataframe.
    '''
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

# Since loading takes a long time we do not want to mess-up the dataframe. So we will first copy it
dff = df.copy()

# Rename some columns for nicer display
dff["time [s]"] = dff["scores.mean_fit_time"]
dff["nodes"] = dff["mean_nodes"]
dff["accuracy"] = 100.0*dff["mean_accuracy"]
dff["size [kb]"] = dff["mean_memory"]

# Filter for 100 MB and check how many experiments are left
dff = dff.loc[dff["size [kb]"] < 100*1024]
print(len(dff))

colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
markers = ["o", "v", "^", "<", ">", "s", "P", "X", "D"]
styles = ["-", "--", "-.", ":","-", "--", "-.", ":","-", "--", "-.", ":",]

aucs = []

# True if we should show plots inside the notebook directly
show = False

for dataset, gdf in dff.groupby("dataset"):
    #gdf = gdf.loc[gdf["size [kb]"] < 0.2*1e6]
    max_kb = None
    for name, group in dff.groupby(["nice_name"]):
        if max_kb is None or group["size [kb]"].max() > max_kb:
            max_kb = group["size [kb]"].max()

    fig = plt.figure()
    for (name, group), marker, color, style in zip(gdf.groupby(["nice_name"]),markers, colors, styles):
        pdf = get_pareto(group, ["accuracy", "size [kb]"])
        pdf = pdf[["nice_name", "accuracy", "size [kb]", "time [s]"]]
        pdf = pdf.sort_values(by=['accuracy'], ascending = True)
        
        x = np.append(pdf["size [kb]"].values, [max_kb])
        y = np.append(pdf["accuracy"].values, [pdf["accuracy"].values[-1]]) / 100.0
        
        x_scatter = np.append(group["size [kb]"].values, [max_kb])
        y_scatter = np.append(group["accuracy"].values,[pdf["accuracy"].values[-1]]) / 100.0

        plt.scatter(x_scatter,y_scatter,s = [2.5**2 for _ in x_scatter], color = color)

        plt.plot(x,y, label=name, color=color) #marker=marker
        aucs.append(
            {
                "model":name,
                #"AUC":np.trapz(y, x),
                "AUC":np.trapz(y, x) / max_kb,
                "dataset":dataset
            }
        )

    print("Dataset {}".format(dataset))
    plt.legend(loc="lower right")
    plt.xlabel("Model Size [KB]")
    plt.ylabel("Accuracy")
    fig.savefig("{}_paretofront.pdf".format(dataset), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

tabledf = pd.DataFrame(aucs)
tabledf.sort_values(by=["dataset","AUC"], inplace = True, ascending=False)
tabledf.to_csv("aucs.csv",index=False)

tabledf.pivot_table(index=["dataset"], values=["AUC"], columns=["model"]).round(4).to_latex("aucs.tex")
display(tabledf.pivot_table(index=["dataset"], values=["AUC"], columns=["model"]).round(4))
