#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import argparse
import random
from scipy.special import softmax
from scipy.io.arff import loadarff

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import river
from river import tree
from river.ensemble import AdaptiveRandomForestClassifier

from JaxModel import JaxModel

from experiment_runner.experiment_runner_v2 import run_experiments, get_ctor_arguments

from BiasedProxEnsemble import BiasedProxEnsemble
from SGDEnsemble import SGDEnsemble
from RiverModel import RiverModel
from PyBiasedProxEnsemble import PyBiasedProxEnsemble

def pre(cfg):
    model_ctor = cfg.pop("model")
    model_params = cfg["model_params"]
    if "out_file" in model_params and model_params["out_file"] is not None:
        model_params["out_file"] = os.path.join(cfg["out_path"], model_params["out_file"])

    model = model_ctor(**model_params)
    return model

def fit(cfg, model):
    model.fit(cfg["X"], cfg["Y"])
    return model

def post(cfg, model):
    return {}
    # i = cfg["run_id"]
    # scores = {}
    # X = cfg["X"]
    # Y = cfg["Y"]

    # def _loss(pred, target):
    #     if "eval_loss" in cfg:
    #         loss_type = cfg["eval_loss"]
    #     else:
    #         loss_type = cfg["loss"]

    #     if loss_type == "mse":
    #         target_one_hot = np.array( [ [1 if y == i else 0 for i in range(model.n_classes_)] for y in target] )
    #         p = softmax(pred, axis=1)
    #         loss = (p - target_one_hot) * (p - target_one_hot)
    #     elif loss_type == "cross-entropy":
    #         target_one_hot = np.array( [ [1 if y == i else 0 for i in range(model.n_classes_)] for y in target] )
    #         p = softmax(pred, axis=1)
    #         loss = -target_one_hot*np.log(p + 1e-7)
    #     else:
    #         raise "Wrong loss given. Loss was {} but expected {{mse, cross-entropy}}".format(loss_type)
    #     return np.mean(loss)

    # train_output = model.predict_proba(X)
    # scores["train_accuracy"] = accuracy_score(Y, train_output.argmax(axis=1))*100.0
    # scores["train_loss"] = _loss(train_output, Y)

    # if isinstance(model, (GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier)):
    #     scores["n_estimators"] = len(model.estimators_)
    #     n_parameters = 0
    #     for est in model.estimators_:
    #         # TODO Add inner nodes
    #         if isinstance(model, GradientBoostingClassifier):
    #             for ei in est:
    #                 n_parameters += model.n_classes_ * ei.tree_.node_count
    #         else:
    #             n_parameters += model.n_classes_ * est.tree_.node_count

    #     scores["n_parameters"] = n_parameters
    # else:
    #     scores["n_estimators"] = model.num_trees()
    #     scores["n_parameters"] = model.num_parameters()

    # return scores

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--single", help="Run via single thread",action="store_true", default=False)
parser.add_argument("-r", "--ray", help="Run via Ray",action="store_true", default=False)
parser.add_argument("-m", "--multi", help="Run via multiprocessing pool",action="store_true", default=False)
parser.add_argument("--ray_head", help="Run via Ray",action="store_true", default="auto")
parser.add_argument("--redis_password", help="Run via Ray",action="store_true", default="5241590000000000")
args = parser.parse_args()

if not args.single and not args.ray and not args.multi:
    print("No processing mode found, defaulting to `single` processing.")
    args.single = True

if args.single:
    basecfg = {
        "out_path":"results/" + datetime.now().strftime('%d-%m-%Y-%H:%M:%S'),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "local",
        "verbose":True
    }
elif args.multi:
    basecfg = {
        "out_path":"results/" + datetime.now().strftime('%d-%m-%Y-%H:%M:%S'),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "multiprocessing",
        "num_cpus":4,
        "verbose":True
    }
else:
    exit(1)

print("LOADING DATA")
data, meta = loadarff("agrawal_a.arff")
#data, meta = loadarff("agrawal_debug.arff")

Xdict = {}
for cname, ctype in zip(meta.names(), meta.types()):
    if cname == "class":
        enc = LabelEncoder()
        Xdict["label"] = enc.fit_transform(data[cname])
    elif ctype == "numeric":
        Xdict[cname] = data[cname]
    else:
        enc = OneHotEncoder(sparse=False)
        tmp = enc.fit_transform(data[cname].reshape(-1, 1))
        for i in range(tmp.shape[1]):
            Xdict[cname + "_" + str(i)] = tmp[:,i]

df = pd.DataFrame(Xdict)
Y = df["label"].values.astype(np.int32)
df = df.drop("label", axis=1)
is_nominal = (df.nunique() == 2).values
nominal_names = [name for nom,name in zip(is_nominal, df.columns.values) if nom ]
# print(nominal_names)

scaler = MinMaxScaler()
X = scaler.fit_transform(df.values.astype(np.float64))

experiment_cfg = {
    "X":X,
    "Y":Y,
    "verbose":True,
    "eval_loss":"cross-entropy",
    "seed":0
}

online_learner_cfg = {
    "n_updates":10000,
    "batch_size":128,
    "out_file":"training.jsonl",
    "sliding_window":True,
    "verbose":args.single,
    "shuffle":False
}

models = []

for T in [16, 32, 64, 128, 256]:
    models.append(
        {
            "model":PyBiasedProxEnsemble,
            "model_params": {
                "loss":"cross-entropy",
                "step_size":1e-2, #1e-3,
                "ensemble_regularizer":"hard-L1",
                "l_ensemble_reg":T,
                "tree_regularizer":None,
                "l_tree_reg":0,
                "normalize_weights":True,
                "init_weight":"average",
                "max_depth":None,
                "scale_batch":0,
                "var_batch":0.001,
                "seed":experiment_cfg["seed"],
                **online_learner_cfg
            },
            **experiment_cfg
        }
    )


models.append(
    {
        "model":RiverModel,
        "model_params": {
            "river_model":river.ensemble.SRPClassifier(
                n_models = 25,
                model = river.tree.HoeffdingTreeClassifier(
                    grace_period = 50,
                    #split_confidence = 1e-6,
                    split_confidence = 0.01,
                    #min_samples_reevaluate = 300,
                    leaf_prediction = "mc",
                    nominal_attributes = nominal_names,
                    max_depth=35
                ),
            ),
            **online_learner_cfg
        },
        **experiment_cfg
    }
)

# for d in [3,5,10,12,15]:
#     models.append(
#         {
#             "model":RiverModel,
#             "river_model":river.ensemble.SRPClassifier(
#                 n_models = 10,
#                 model = river.tree.HoeffdingTreeClassifier(
#                     grace_period = 50,
#                     #split_confidence = 1e-6,
#                     split_confidence = 0.01,
#                     #min_samples_reevaluate = 300,
#                     leaf_prediction = "mc",
#                     nominal_attributes = nominal_names,
#                     max_depth=20
#                 ),
#             ),
#             **shared_cfg
#         }
#     )

# for T in [256]:
#     models.append(
#         {
#             "model":SGDEnsemble,
#             "max_trees":T,
#             "init_mode":"fully-random",
#             "next_mode":"gradient",
#             "is_nominal":is_nominal,
#             **shared_cfg,
#             **grad_cfg
#         }
#     )

# for l_reg in [1e-2,1e-3,5e-1,5e-2,5e-3]:
#     models.append(
#             {
#                 "model":BiasedProxEnsemble,
#                 "max_trees":0,
#                 "l_reg":l_reg,
#                 "init_mode":"train",
#                 "next_mode":"gradient",
#                 "is_nominal":is_nominal,
#                 **shared_cfg,
#                 **grad_cfg
#             }
#         )

# random.shuffle(models)

run_experiments(basecfg, models)
