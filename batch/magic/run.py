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

from experiment_runner.experiment_runner_v2 import run_experiments, get_ctor_arguments

from JaxModel import JaxModel
from ProxPruningClassifier import ProxPruningClassifier
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
        elif key == "x_test":
            X = cfg["X"]
            i = cfg["run_id"]
            _, itest = cfg["idx"][i]
            expected["x_test"] = X[itest]
        elif key == "y_test":
            Y = cfg["Y"]
            i = cfg["run_id"]
            _, itest = cfg["idx"][i]
            expected["y_test"] = Y[itest]
        
        if key in tmpcfg:
            expected[key] = tmpcfg[key]
    
    model = model_ctor(**expected)
    return model

def fit(cfg, model):
    i = cfg["run_id"]
    itrain, _ = cfg["idx"][i]
    X, Y = cfg["X"],cfg["Y"]

    model.fit(X[itrain], Y[itrain])
    return model

def post(cfg, model):
    i = cfg["run_id"]
    itrain, itest = cfg["idx"][i]
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

    test_output = model.predict_proba(X[itest])
    scores["test_accuracy"] = accuracy_score(Y[itest], test_output.argmax(axis=1))*100.0
    scores["test_loss"] = _loss(test_output, Y[itest])

    train_output = model.predict_proba(X[itrain])
    scores["train_accuracy"] = accuracy_score(Y[itrain], train_output.argmax(axis=1))*100.0
    scores["train_loss"] = _loss(train_output, Y[itrain])

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
        "num_cpus":7,
        "verbose":True
    }
else:
    exit(1)

df = pd.read_csv("magic04.data")
X = df.values[:,:-1].astype(np.float64)
Y = df.values[:,-1]
Y = np.array([0 if y == 'g' else 1 for y in Y])

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

n_splits = 5
kf = KFold(n_splits=n_splits, random_state=12345, shuffle=True)
idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X)], dtype=object)

shared_cfg = {
    "n_updates":5000,
    "verbose":True,
    "eval_every_items":4096,
    "eval_every_epochs":1,
    "X":X,
    "Y":Y,
    "idx":idx,
    "repetitions":n_splits,
    "seed":12345,
    "batch_size":512,
    "loss":"hinge2"
}

grad_cfg = {
    "step_size":1e-4,
    "init_weight":1.0/256 #1.0/1024.0
}

models = []

models.append(
    {
        "model":ProxPruningClassifier,
        "l_reg":0,#1e-4,
        "base_estimator": RandomForestClassifier(bootstrap=True, max_depth=15, n_estimators=256),
        "n_jobs":1,
        **shared_cfg,
        **grad_cfg
    }
)

'''
for T in [256]:
    models.append(
        {
            "model":SGDEnsemble,
            "max_trees":T,
            "init_mode":"train",
            "next_mode":"gradient",
            "max_depth":15,
            **shared_cfg,
            **grad_cfg
        }
    )

for l_reg in [1e-2,1e-3,5e-1,5e-2,5e-3]:
    models.append(
            {
                "model":BiasedProxEnsemble,
                "max_trees":0,
                "l_reg":l_reg,
                "init_mode":"train",
                "next_mode":"gradient",
                "max_depth":15,
                **shared_cfg,
                **grad_cfg
            }
        )


for init_mode in ["train", "fully-random"]:
    for next_mode in ["gradient", "incremental"]:
        for d in [2,5,7,10]:
            for l_reg in [1e-1,1e-2,1e-3,5e-1,5e-2,5e-3]:
                models.append(
                    {
                        "model":BiasedProxEnsemble,
                        "max_trees":0,
                        "l_reg":l_reg,
                        "init_mode":init_mode,
                        "next_mode":next_mode,
                        "max_depth":d,
                        **shared_cfg,
                        **grad_cfg
                    }
                )

            for T in [16, 32, 64, 128, 256]:
                models.append(
                    {
                        "model":SGDEnsemble,
                        "max_trees":T,
                        "init_mode":init_mode,
                        "next_mode":next_mode,
                        "max_depth":d,
                        **shared_cfg,
                        **grad_cfg
                    }
                )

for T in [1,2,5]:
    for ts in [1.0, 2.0]:
        models.append (
            {
                "model":JaxModel,
                "n_trees":T,
                "temp_scaling":ts,
                **shared_cfg,
                **grad_cfg
            }
        )

for T in [16, 32, 64,128,256]:
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
            **shared_cfg
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
            **shared_cfg
        }
    )


for T in [16, 32, 64, 128, 256]:
    models.append(
        {
            "model":RandomForestClassifier,
            "bootstrap":True,
            "max_depth":15,
            "n_estimators":T,
            #"max_samples":shared_cfg["batch_size"],
            **shared_cfg,
        }
    )

    models.append(
        {
            "model":ExtraTreesClassifier,
            "bootstrap":True,
            "max_depth":15,
            "n_estimators":T,
            #"max_samples":shared_cfg["batch_size"],
            **shared_cfg,
        }
    )

    tmp_cfg = shared_cfg.copy()
    tmp_cfg.pop("loss")
    models.append(
        {
            "model":GradientBoostingClassifier,
            "n_estimators":T,
            "max_depth":15,
            "loss":"deviance",
            "eval_loss":"cross-entropy",
            **tmp_cfg,
            #"subsample":int(shared_cfg["batch_size"]) / min([len(i[0]) for i in idx])
        }
    )
'''

random.shuffle(models)

run_experiments(basecfg, models)
