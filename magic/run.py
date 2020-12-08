#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import argparse
from scipy.special import softmax

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
        if key in tmpcfg:
            if key == "x_test":
                X = cfg["X"]
                i = cfg["run_id"]
                _, itest = cfg["idx"][i]
                expected["x_test"] = X[itest]
            elif key == "y_test":
                Y = cfg["Y"]
                i = cfg["run_id"]
                _, itest = cfg["idx"][i]
                expected["x_test"] = Y[itest]
            elif key == "out_file":
                expected["out_file"] = os.path.join(cfg["out_path"], "training.jsonl")
            else:
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
            loss = (pred - target_one_hot) * (pred - target_one_hot)
        elif loss_type == "cross-entropy":
            target_one_hot = np.array( [ [1 if y == i else 0 for i in range(model.n_classes_)] for y in target] )
            p = softmax(pred, axis=1)
            loss = -target_one_hot*np.log(p)
        else:
            raise "Wrong loss given. Loss was {} but expected {mse, cross-entropy}".format(loss_type)
        return np.mean(loss)


    test_output = model.predict_proba(X[itest])
    scores["test_accuracy"] = accuracy_score(Y[itest], test_output.argmax(axis=1))*100.0
    scores["test_loss"] = _loss(test_output, Y[itest])

    train_output = model.predict_proba(X[itrain])
    scores["train_accuracy"] = accuracy_score(Y[itrain], train_output.argmax(axis=1))*100.0
    scores["train_loss"] = _loss(train_output, Y[itrain])

    if isinstance(model, (GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier)):
        scores["n_estimators"] = len(model.estimators_)
        n_nodes = 0
        for est in model.estimators_:
            if isinstance(model, GradientBoostingClassifier):
                for ei in est:
                    n_nodes += ei.tree_.node_count
            else:
                n_nodes += est.tree_.node_count

        scores["n_nodes"] = n_nodes
    else:
        scores["n_estimators"] = model.num_trees()
        scores["n_nodes"] = model.num_nodes()

    return scores

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local", help="Run on local machine",action="store_true", default=False)
parser.add_argument("-r", "--ray", help="Run via Ray",action="store_true", default=False)
parser.add_argument("--ray_head", help="Run via Ray",action="store_true", default="auto")
parser.add_argument("--redis_password", help="Run via Ray",action="store_true", default="5241590000000000")
args = parser.parse_args()

if (args.local and args.ray) or (not args.local and not args.ray):
    print("Either you specified to use both, ray _and_ local mode or you specified to use none of both. Please choose either. Defaulting to `local` processing.")
    args.local = True

# df = pd.read_csv("magic/magic04.data")
df = pd.read_csv("magic04.data")
X = df.values[:,:-1].astype(np.float64)
Y = df.values[:,-1]
Y = np.array([0 if y == 'g' else 1 for y in Y])

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

n_splits = 5
kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X)])

if args.local:
    basecfg = {
        "out_path":"results/" + datetime.now().strftime('%d-%m-%Y-%H:%M:%S'),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "local",
        "verbose":True
    }
else:
    pass
    # basecfg = {
    #     "out_path":os.path.join("/data/d3/buschjae/gncl/imagenet", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
    #     #"out_path":"/data/d3/buschjae/gncl/imagenet/12-11-2020-14:41:16",
    #     "pre": pre,
    #     "post": post,
    #     "fit": fit,
    #     "backend": "ray",
    #     "ray_head": args.ray_head,
    #     "redis_password": args.redis_password,
    #     "verbose":False
    # }

models = []


shared_cfg = {
    "max_depth":5,
    "loss":"mse",
    "batch_size":32,
    "epochs":50,
    "verbose":True,
    "eval_every_items":512,
    "eval_every_epochs":1,
    "X":X,
    "Y":Y,
    "idx":idx,
    "repetitions":n_splits,
    "seed":12345
}

for T in [32, 64, 128, 256]:
    models.append(
        {
            "model":RandomForestClassifier,
            "bootstrap":True,
            "max_samples":shared_cfg["batch_size"],
            "max_depth":shared_cfg["max_depth"],
            "loss":shared_cfg["loss"],
            "n_estimators":T,
            "verbose":False,
            "X":X,
            "Y":Y,
            "idx":idx,
            "repetitions":n_splits,
            "random_state":shared_cfg["seed"]
        }
    )

    models.append(
        {
            "model":ExtraTreesClassifier,
            "bootstrap":True,
            "max_samples":shared_cfg["batch_size"],
            "max_depth":shared_cfg["max_depth"],
            "loss":shared_cfg["loss"],
            "n_estimators":T,
            "verbose":False,
            "X":X,
            "Y":Y,
            "idx":idx,
            "repetitions":n_splits,
            "random_state":shared_cfg["seed"]
        }
    )

    models.append(
        {
            "model":GradientBoostingClassifier,
            "n_estimators":T,
            "max_depth":shared_cfg["max_depth"],
            "loss":"deviance",
            "eval_loss":shared_cfg["loss"],
            "subsample":shared_cfg["batch_size"] / min([len(i[0]) for i in idx]),
            "verbose":False,
            "X":X,
            "Y":Y,
            "idx":idx,
            "repetitions":n_splits,
            "random_state":shared_cfg["seed"]
        }
    )

    models.append(
        {
            "model":SGDEnsemble,
            "max_trees":T,
            "step_size":0.5,
            "l_reg":6e-2,
            "mode":"random",
            "init_weight":1.0,
            **shared_cfg
        }
    )

    models.append(
        {
            "model":SGDEnsemble,
            "max_trees":T,
            "step_size":0.5,
            "l_reg":6e-2,
            "mode":"trained",
            "init_weight":1.0,
            **shared_cfg
        }
    )

    models.append(
        {
            "model":SGDEnsemble,
            "max_trees":T,
            "step_size":0.5,
            "l_reg":6e-2,
            "mode":"fully-random",
            "init_weight":1.0,
            **shared_cfg
        }
    )

    models.append(
        {
            "model":RiverModel,
            "river_model":river.ensemble.AdaBoostClassifier(
                model = tree.ExtremelyFastDecisionTreeClassifier(
                    grace_period=200,
                    split_confidence=1e-5,
                    min_samples_reevaluate=100
                ),
                n_models = T,
                seed = shared_cfg["seed"]
            )
            **shared_cfg
        }
    )

    models.append(
        {
            "model":RiverModel,
            "river_model":river.ensemble.BaggingClassifier(
                model = tree.ExtremelyFastDecisionTreeClassifier(
                    grace_period=200,
                    split_confidence=1e-5,
                    min_samples_reevaluate=100
                ),
                n_models = T,
                seed = shared_cfg["seed"]
            )
            **shared_cfg
        }
    )

    models.append (
        {
            "model":JaxModel,
            "step_size":1e-1,
            "n_trees":T
            **shared_cfg
        }
    )

for l in [ 2e-2, 4e-2, 6e-2]:
    models.append(
        {
            "model":BiasedProxEnsemble,
            "max_trees":0,
            "step_size":0.25,
            "l_reg":l,
            "mode":"random",
            "init_weight":1.0,
            **shared_cfg
        }
    )

    models.append(
        {
            "model":BiasedProxEnsemble,
            "max_trees":0,
            "step_size":0.25,
            "l_reg":l,
            "mode":"trained",
            "init_weight":1.0,
            **shared_cfg
        }
    )

    models.append(
        {
            "model":BiasedProxEnsemble,
            "max_trees":0,
            "step_size":0.25,
            "l_reg":l,
            "mode":"fully-random",
            "init_weight":1.0,
            **shared_cfg
        }
    )


run_experiments(basecfg, models)
