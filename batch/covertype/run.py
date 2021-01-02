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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import river
from river import tree
from river.ensemble import AdaptiveRandomForestClassifier

from JaxModel import JaxModel

from experiment_runner.experiment_runner_v2 import run_experiments, get_ctor_arguments

sys.path.append("../")
from BiasedProxEnsemble import BiasedProxEnsemble
from SGDEnsemble import SGDEnsemble
from RiverModel import RiverModel

def pre(cfg):
    model_ctor = cfg.pop("model")
    tmpcfg = cfg
    expected = {}
    for key in get_ctor_arguments(model_ctor):
        if key == "out_file":
            expected["out_file"] = os.path.join(cfg["out_path"], "training.jsonl")
        # elif key == "x_test":
        #     X = cfg["X"]
        #     i = cfg["run_id"]
        #     _, itest = cfg["idx"][i]
        #     expected["x_test"] = X[itest]
        # elif key == "y_test":
        #     Y = cfg["Y"]
        #     i = cfg["run_id"]
        #     _, itest = cfg["idx"][i]
        #     expected["y_test"] = Y[itest]
        
        if key in tmpcfg:
            expected[key] = tmpcfg[key]
    
    model = model_ctor(**expected)
    return model

def fit(cfg, model):
    # i = cfg["run_id"]
    # itrain, _ = cfg["idx"][i]
    # X, Y = cfg["X"],cfg["Y"]

    model.fit(cfg["X"], cfg["Y"])
    return model

def post(cfg, model):
    i = cfg["run_id"]
    # itrain, itest = cfg["idx"][i]
    scores = {}
    X = cfg["X"]
    Y = cfg["Y"]

    def _loss(pred, target):
        if "eval_loss" in cfg:
            loss_type = cfg["eval_loss"]
        else:
            loss_type = cfg["loss"]

        if loss_type == "mse":
            target_one_hot = np.array( [ [1 if y == i else 0 for i in range(model.n_classes_)] for y in target] )
            p = softmax(pred, axis=1)
            loss = (p - target_one_hot) * (p - target_one_hot)
        elif loss_type == "cross-entropy":
            target_one_hot = np.array( [ [1 if y == i else 0 for i in range(model.n_classes_)] for y in target] )
            p = softmax(pred, axis=1)
            loss = -target_one_hot*np.log(p + 1e-7)
        else:
            raise "Wrong loss given. Loss was {} but expected {{mse, cross-entropy}}".format(loss_type)
        return np.mean(loss)

    # test_output = model.predict_proba(X)
    # scores["test_accuracy"] = accuracy_score(Y, test_output.argmax(axis=1))*100.0
    # scores["test_loss"] = _loss(test_output, Y)

    train_output = model.predict_proba(X)
    scores["train_accuracy"] = accuracy_score(Y, train_output.argmax(axis=1))*100.0
    scores["train_loss"] = _loss(train_output, Y)

    if isinstance(model, (GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier)):
        scores["n_estimators"] = len(model.estimators_)
        n_parameters = 0
        for est in model.estimators_:
            # TODO Add inner nodes
            if isinstance(model, GradientBoostingClassifier):
                for ei in est:
                    n_parameters += model.n_classes_ * ei.tree_.node_count
            else:
                n_parameters += model.n_classes_ * est.tree_.node_count

        scores["n_parameters"] = n_parameters
    else:
        scores["n_estimators"] = model.num_trees()
        scores["n_parameters"] = model.num_parameters()

    return scores

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local", help="Run on local machine",action="store_true", default=False)
parser.add_argument("-r", "--ray", help="Run via Ray",action="store_true", default=False)
parser.add_argument("-m", "--multi", help="Run via multiprocessing pool",action="store_true", default=False)
parser.add_argument("--ray_head", help="Run via Ray",action="store_true", default="auto")
parser.add_argument("--redis_password", help="Run via Ray",action="store_true", default="5241590000000000")
args = parser.parse_args()

if not args.local and not args.ray and not args.multi:
    print("No processing mode found, defaulting to `local` processing.")
    args.local = True

df = pd.read_csv("covtype.data")
X = df.values[:,:-1].astype(np.float64)
Y = df.values[:,-1]
Y = Y - min(Y)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

if args.local:
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

shared_cfg = {
    "max_depth":5,
    "loss":"cross-entropy",
    "batch_size":128,
    #"epochs":int(len(X)/256),
    # "epochs":1,
    "n_updates":10000,
    "verbose":True,
    "eval_every_items":4096,
    "eval_every_epochs":1,
    "X":X,
    "Y":Y,
    "seed":12345,
    "step_size":1e-2,
    "init_weight":1.0
}

models = []

models.append(
    {
        "model":SGDEnsemble,
        "max_trees":256,
        "init_mode":"fully-random",
        "next_mode":"gradient",
        **shared_cfg
    }
)

'''
for init_mode in ["train", "fully-random"]:
    for next_mode in ["gradient", "incremental"]:
        for l_reg in [1e-1,1e-2,1e-3,5e-1,5e-2,5e-3]:
            models.append(
                {
                    "model":BiasedProxEnsemble,
                    "max_trees":0,
                    "l_reg":l_reg,
                    "init_mode":init_mode,
                    "next_mode":next_mode,
                    **shared_cfg
                }
            )

        for T in [16, 32, 64,128,256]:
            models.append(
                {
                    "model":SGDEnsemble,
                    "max_trees":T,
                    "init_mode":init_mode,
                    "next_mode":next_mode,
                    **shared_cfg
                }
            )

for T in [1,2,5]:
    for ts in [1.0, 2.0]:
        models.append (
            {
                "model":JaxModel,
                "n_trees":T,
                "temp_scaling":ts,
                **shared_cfg
            }
        )

for T in [16, 32,64]:
    models.append(
        {
            "model":RiverModel,
            "river_model":river.ensemble.SRPClassifier(
                n_models = 2,
                model = river.tree.ExtremelyFastDecisionTreeClassifier(
                    grace_period = 300,
                    split_confidence = 1e-6,
                    min_samples_reevaluate = 300,
                    leaf_prediction = "mc"
                ),
            ),
            "loss":"cross-entropy",
            "batch_size":1,
            "verbose":False,
            "epochs":1,
            "eval_every_items":2048,
            "eval_every_epochs":1,
            "X":X,
            "Y":Y,
            "seed":12345
        }
    )

    models.append(
        {
            "model":RiverModel,
            "river_model":river.ensemble.BaggingClassifier(
                n_models = T,
                model = river.tree.ExtremelyFastDecisionTreeClassifier(
                    grace_period = 300,
                    split_confidence = 1e-6,
                    min_samples_reevaluate = 300,
                    leaf_prediction = "mc"
                ),
            ),
            "loss":"cross-entropy",
            "batch_size":1,
            "verbose":False,
            "epochs":1,
            "eval_every_items":2048,
            "eval_every_epochs":1,
            "X":X,
            "Y":Y,
            "seed":12345
        }
    )
'''

# for T in [16, 32, 64, 128]:
#     for bs in [32, 128, 512, 1024]:
#         models.append(
#             {
#                 "model":RandomForestClassifier,
#                 "bootstrap":True,
#                 "max_samples":bs,
#                 "max_depth":None,
#                 "loss":"cross-entropy",
#                 "n_estimators":T,
#                 "verbose":False,
#                 "X":X,
#                 "Y":Y,
#                 "random_state":12345
#             }
#         )

#         models.append(
#             {
#                 "model":ExtraTreesClassifier,
#                 "bootstrap":True,
#                 "max_samples":bs,
#                 "max_depth":None,
#                 "loss":"cross-entropy",
#                 "n_estimators":T,
#                 "verbose":False,
#                 "X":X,
#                 "Y":Y,
#                 "idx":idx,
#                 "repetitions":n_splits,
#                 "random_state":12345
#             }
#         )

#         models.append(
#             {
#                 "model":GradientBoostingClassifier,
#                 "n_estimators":T,
#                 "max_depth":None,
#                 "loss":"deviance",
#                 "eval_loss":"cross-entropy",
#                 "subsample":bs / min([len(i[0]) for i in idx]),
#                 "verbose":False,
#                 "X":X,
#                 "Y":Y,
#                 "idx":idx,
#                 "repetitions":n_splits,
#                 "random_state":12345
#             }
#         )

random.shuffle(models)

run_experiments(basecfg, models)
