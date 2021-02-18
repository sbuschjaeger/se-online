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
    if "out_path" in model_params and model_params["out_path"] is not None:
        model_params["out_path"] = cfg["out_path"]

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
        "backend": "single",
        "verbose":True
    }
elif args.multi:
    basecfg = {
        "out_path":"results/" + datetime.now().strftime('%d-%m-%Y-%H:%M:%S'),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "multiprocessing",
        "num_cpus":9,
        "verbose":True
    }
else:
    exit(1)

print("Loading data")
df = pd.reads_csv("cs-training.csv")
df = df.dropna()

Y = df["SeriousDlqin2yrs"].values.astype(np.int32)
print(set(Y))
df = df.drop(["SeriousDlqin2yrs"], axis=1)
scaler = MinMaxScaler()
X = scaler.fit_transform(df.values.astype(np.float64))
print(X.shape)
adwd
experiment_cfg = {
    "X":X,
    "Y":Y,
    "verbose":True,
    "eval_loss":"cross-entropy",
    "seed":0
}

online_learner_cfg = {
    "epochs":1,
    "batch_size":16, 
    "out_path":".",
    "sliding_window":True,
    "verbose":args.single,
    "shuffle":False,
    "eval_every_epochs":1
}

models = []

'''
for bs in [8,16,32]:
    online_learner_cfg["batch_size"] = bs

    for d in [2,4,6]:
        for T in [128]:
            for lr in [1e-1, 5e-1, 1, 2, 3, 4]:
                models.append(
                    {
                        "model":PyBiasedProxEnsemble,
                        "model_params": {
                            #"loss":"cross-entropy",
                            "loss":"mse",
                            "step_size":lr, #1e-3,
                            "ensemble_regularizer":"hard-L1",
                            "l_ensemble_reg":T,
                            "tree_regularizer":None,
                            "l_tree_reg":0,
                            "normalize_weights":True,
                            "init_weight":"average",
                            "max_depth":d,
                            "seed":experiment_cfg["seed"],
                            "update_trees":True,
                            **online_learner_cfg
                        },
                        **experiment_cfg
                    }
                )

                models.append(
                    {
                        "model":PyBiasedProxEnsemble,
                        "model_params": {
                            "loss":"cross-entropy",
                            "step_size":lr, #1e-3,
                            "ensemble_regularizer":"hard-L1",
                            "l_ensemble_reg":T,
                            "tree_regularizer":None,
                            "l_tree_reg":0,
                            "normalize_weights":True,
                            "init_weight":"average",
                            "max_depth":d,
                            "seed":experiment_cfg["seed"],
                            "update_trees":False,
                            **online_learner_cfg
                        },
                        **experiment_cfg
                    }
                )
'''

online_learner_cfg["batch_size"] = 1
for lp in ["nba","mc"]:
    for gp in [10,50,100,500]:
        for sc in [1e-2,1e-3,1e-4]:
            models.append(
                {
                    "model":RiverModel,
                    "model_params": {
                        "river_model":river.tree.HoeffdingTreeClassifier(
                            grace_period = gp,
                            split_confidence = sc,
                            leaf_prediction = lp
                        ),
                        **online_learner_cfg
                    },
                    # TODO This is currently redundant any only required to distinguish the results in the end. Can we change that?
                    "river_model_params" : {
                        "grace_period" : gp,
                        "split_confidence" : sc,
                        "leaf_prediction" : lp
                    },
                    **experiment_cfg
                }
            )

            models.append(
                {
                    "model":RiverModel,
                    "model_params": {
                        "river_model":river.tree.ExtremelyFastDecisionTreeClassifier(
                            grace_period = gp,
                            split_confidence = sc,
                            leaf_prediction = lp
                        ),
                        **online_learner_cfg
                    },
                    "river_model_params" : {
                        "grace_period" : gp,
                        "split_confidence" : sc,
                        "leaf_prediction" : lp
                    },
                    **experiment_cfg
                }
            )

            for T in [5,10,25]:
                models.append(
                    {
                        "model":RiverModel,
                        "model_params": {
                            "river_model":river.ensemble.SRPClassifier(
                                n_models = T,
                                model = river.tree.HoeffdingTreeClassifier(
                                    grace_period = gp,
                                    split_confidence = sc,
                                    leaf_prediction = lp
                                )
                            ),
                            **online_learner_cfg
                        },
                        "river_model_params" : {
                            "grace_period" : gp,
                            "split_confidence" : sc,
                            "leaf_prediction" : lp
                        },
                        **experiment_cfg
                    }
                )

# online_learner_cfg["batch_size"] = 32
# for d in [2,4,6]:
#     models.append(
#         {
#             "model":JaxModel,
#             "model_params": {
#                 "loss":"cross-entropy",
#                 "step_size":1e-2,
#                 "max_depth":d,
#                 "n_trees":1,
#                 **online_learner_cfg
#             },
#             **experiment_cfg
#         }
#     )

#     models.append(
#         {
#             "model":JaxModel,
#             "model_params": {
#                 "loss":"cross-entropy",
#                 "step_size":1e-2,
#                 "max_depth":d,
#                 "n_trees":10,
#                 **online_learner_cfg
#             },
#             **experiment_cfg
#         }
#     )


random.shuffle(models)

run_experiments(basecfg, models)
