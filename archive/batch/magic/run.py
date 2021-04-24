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
from sklearn.ensemble import AdaBoostClassifier
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
from PyBiasedProxEnsemble import PyBiasedProxEnsemble

def pre(cfg):
    model_ctor = cfg.pop("model")
    model_params = cfg["model_params"]
    if "out_file" in model_params and model_params["out_file"] is not None:
        model_params["out_file"] = os.path.join(cfg["out_path"], model_params["out_file"])

    if "base_ensemble" in cfg:
        base = cfg["base_ensemble"]
        i = cfg["run_id"]
        itrain, _ = cfg["idx"][i]
        X, Y = cfg["X"],cfg["Y"]
        base.fit(X, Y)
        model_params["estimators"] = base.estimators_

    model = model_ctor(**model_params)
        
    # tmpcfg = cfg
    # expected = {}
    # for key in get_ctor_arguments(model_ctor):
    #     if key == "out_file":
    #         expected["out_file"] = os.path.join(cfg["out_path"], "training.jsonl")
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
        
    #     if key in tmpcfg:
    #         expected[key] = tmpcfg[key]
    
    # model = model_ctor(**expected)
    return model

def fit(cfg, model):
    i = cfg["run_id"]
    itrain, _ = cfg["idx"][i]
    X, Y = cfg["X"],cfg["Y"]

    model.fit(X[itrain], Y[itrain])

    # if (isinstance(model, RandomForestClassifier)):
    #     model_ctor = PyBiasedProxEnsemble
    #     tmpcfg = cfg
    #     expected = {}
    #     for key in get_ctor_arguments(model_ctor):
    #         if key == "out_file":
    #             expected["out_file"] = os.path.join(cfg["out_path"], "training.jsonl")
    #         if key in tmpcfg:
    #             expected[key] = tmpcfg[key]
        
    #     tmp_model = model_ctor(**expected)
    #     tmp_model.fit(X[itrain], Y[itrain])
    #     model.estimators_ = tmp_model.estimators_

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
            loss = -target_one_hot*np.log(p + 1e-7)
        elif loss_type == "hinge2":
            target_one_hot = np.array( [ [1.0 if y == i else -1.0 for i in range(model.n_classes_)] for y in target] )
            zeros = np.zeros_like(target_one_hot)
            loss = np.maximum(1.0 - target_one_hot * pred, zeros)**2
        else:
            raise "Wrong loss given. Loss was {} but expected {{mse, cross-entropy}}".format(loss_type)
        return np.mean(loss)

    test_output = model.predict_proba(X[itest])
    scores["test_accuracy"] = accuracy_score(Y[itest], test_output.argmax(axis=1))*100.0
    scores["test_loss"] = _loss(test_output, Y[itest])

    train_output = model.predict_proba(X[itrain])
    scores["train_accuracy"] = accuracy_score(Y[itrain], train_output.argmax(axis=1))*100.0
    scores["train_loss"] = _loss(train_output, Y[itrain])

    if isinstance(model, (GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, PyBiasedProxEnsemble)):
        scores["n_estimators"] = len(model.estimators_)
        n_parameters = 0
        for est in model.estimators_:
            # TODO Add inner nodes
            if isinstance(model, GradientBoostingClassifier):
                for ei in est:
                    n_parameters += ei.tree_.node_count
            else:
                n_parameters += est.tree_.node_count

        scores["n_parameters"] = n_parameters
    else:
        scores["n_estimators"] = model.num_trees()
        scores["n_parameters"] = model.num_parameters()

    return scores

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

experiment_cfg = {
    "X":X,
    "Y":Y,
    "idx":idx,
    "repetitions":n_splits,
    "loss":"mse",
    "seed":0
}

online_learner_cfg = {
    "verbose":args.single,
    "n_updates":1000,
    "eval_every_items":4096,
    "eval_every_epochs":1,
    "batch_size":4096,
    "out_file":"training.jsonl"
}

models = []

for T in [16, 32, 64, 128, 256]:
    models.append(
        {
            "model":PyBiasedProxEnsemble,
            "model_params": {
                "loss":experiment_cfg["loss"],
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
                "sliding_window":False,
                **online_learner_cfg
            },
            **experiment_cfg
        }
    )

    models.append(
        {
            "model":RandomForestClassifier,
            "model_params": {
                "bootstrap":True,
                "max_depth":None,
                "n_estimators":T,
                #"max_samples":shared_cfg["batch_size"],
            },
            **experiment_cfg,
        }
    )

    models.append(
        {
            "model":ExtraTreesClassifier,
            "model_params": {
                "bootstrap":True,
                "max_depth":None,
                "n_estimators":T,
                #"max_samples":shared_cfg["batch_size"],
            },
            **experiment_cfg,
        }
    )

    tmp_cfg = experiment_cfg.copy()
    tmp_cfg.pop("loss")
    models.append(
        {
            "model":GradientBoostingClassifier,
            "model_params": {
                "n_estimators":T,
                "max_depth":15,
                "loss":"deviance",
            },
            "eval_loss":"cross-entropy",
            **tmp_cfg
            #"subsample":int(shared_cfg["batch_size"]) / min([len(i[0]) for i in idx])
        }
    )

# random.shuffle(models)

run_experiments(basecfg, models)
