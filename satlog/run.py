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
    X, Y = cfg["x_train"],cfg["y_train"]

    model.fit(X, Y)
    return model

def post(cfg, model):
    X_test = cfg["x_test"]
    Y_test = cfg["y_test"]

    X_train = cfg["x_train"]
    Y_train = cfg["y_train"]
    scores = {}

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

    test_output = model.predict_proba(X_test)
    scores["test_accuracy"] = accuracy_score(Y_test, test_output.argmax(axis=1))*100.0
    scores["test_loss"] = _loss(test_output, Y_test)

    train_output = model.predict_proba(X_train)
    scores["train_accuracy"] = accuracy_score(Y_train, train_output.argmax(axis=1))*100.0
    scores["train_loss"] = _loss(train_output, Y_train)

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

df_train = pd.read_csv("sat.trn", sep=" ", header=None)
X_train = df_train.values[:,:-1].astype(np.float64)
Y_train = df_train.values[:,-1]
# Y_train = Y_train - min(Y_train)
Y_train[Y_train == 7] = 0

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

df_test = pd.read_csv("sat.tst", sep=" ", header=None)
X_test = df_test.values[:,:-1].astype(np.float64)
Y_test = df_test.values[:,-1]
X_test = scaler.transform(X_test)
# Y_test = Y_test - min(Y_test)
Y_test[Y_test == 7] = 0
print(set(Y_test))
print(set(Y_train))

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

shared_cfg = {
    "max_depth":12,
    "loss":"cross-entropy",
    "batch_size":256,
    "epochs":2,
    "verbose":True,
    "eval_every_items":0,
    "eval_every_epochs":1,
    "repetitions":0,
    "seed":12345
}

models = []



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
        "batch_size":128,
        "verbose":True,
        "epochs":50,
        "eval_every_items":0,
        "eval_every_epochs":1,
        "x_train":X_train,
        "x_test":X_test,
        "y_train":Y_train,
        "y_test":Y_test,
        "repetitions":1,
        "seed":12345
    }
)

# models.append(
#     {
#         "model":BiasedProxEnsemble,
#         "max_trees":0,
#         "step_size":1e-2,
#         "l_reg":1e-3,
#         "init_mode":"train",
#         "init_weight":1.0,
#         "next_mode":"gradient",
#         **shared_cfg
#     }
# )

'''
for bs, depth in zip([32,128,512,2048], [3,5,7,10]):
    shared_cfg = {
        "max_depth":depth,
        "loss":"cross-entropy",
        "batch_size":bs,
        "epochs":10,
        "verbose":False,
        "eval_every_items":2048,
        "eval_every_epochs":1,
        "X":X,
        "Y":Y,
        "idx":idx,
        "repetitions":n_splits,
        "seed":12345
    }

    for s in [1e-2,1e-1]:
        for l in [1e-2, 1e-3, 1e-4]:
            models.append(
                {
                    "model":BiasedProxEnsemble,
                    "max_trees":0,
                    "step_size":s,
                    "l_reg":l,
                    "init_mode":"train",
                    "init_weight":1.0,
                    "next_mode":"gradient",
                    **shared_cfg
                }
            )

            models.append(
                {
                   "model":BiasedProxEnsemble,
                    "max_trees":0,
                    "step_size":s,
                    "l_reg":l,
                    "init_mode":"fully-random",
                    "init_weight":1.0,
                    "next_mode":"gradient",
                    **shared_cfg
                }
            )

            models.append(
                {
                    "model":BiasedProxEnsemble,
                    "max_trees":128,
                    "step_size":s,
                    "l_reg":l,
                    "init_mode":"train",
                    "init_weight":1.0,
                    "next_mode":"gradient",
                    **shared_cfg
                }
            )

            models.append(
                {
                    "model":BiasedProxEnsemble,
                    "max_trees":128,
                    "step_size":s,
                    "l_reg":l,
                    "init_mode":"fully-random",
                    "init_weight":1.0,
                    "next_mode":"gradient",
                    **shared_cfg
                }
            )

for bs, depth in zip([32,32,128,512,1024], [1,3,5,7,10]):
    shared_cfg = {
        "max_depth":depth,
        "loss":"cross-entropy",
        "batch_size":bs,
        "epochs":10,
        "verbose":False,
        "eval_every_items":2048,
        "eval_every_epochs":1,
        "X":X,
        "Y":Y,
        "idx":idx,
        "repetitions":n_splits,
        "seed":12345
    }

    for T in [16, 32, 64]:
        models.append(
            {
                "model":SGDEnsemble,
                "max_trees":T,
                "step_size":1e-1,
                "init_mode":"fully-random",
                "init_weight":1.0,
                "next_mode":"gradient",
                **shared_cfg
            }
        )

        models.append(
            {
                "model":SGDEnsemble,
                "max_trees":T,
                "step_size":1e-1,
                "init_mode":"train",
                "init_weight":1.0,
                "next_mode":"gradient",
                **shared_cfg
            }
        )

        models.append(
            {
                "model":SGDEnsemble,
                "max_trees":T,
                "step_size":1e-1,
                "init_mode":"random",
                "init_weight":1.0,
                "next_mode":"gradient",
                **shared_cfg
            }
        )

        models.append(
            {
                "model":SGDEnsemble,
                "max_trees":T,
                "step_size":1e-1,
                "init_mode":"random",
                "init_weight":1.0,
                "next_mode":"none",
                **shared_cfg
            }
        )

        models.append(
            {
                "model":SGDEnsemble,
                "max_trees":T,
                "step_size":1e-1,
                "init_mode":"fully-random",
                "init_weight":1.0,
                "next_mode":"none",
                **shared_cfg
            }
        )

        models.append(
            {
                "model":SGDEnsemble,
                "max_trees":T,
                "step_size":1e-1,
                "init_mode":"train",
                "init_weight":1.0,
                "next_mode":"none",
                **shared_cfg
            }
        )

for bs, depth in zip([32, 32, 128], [1, 3, 5]):
    shared_cfg = {
        "max_depth":depth,
        "loss":"cross-entropy",
        "batch_size":bs,
        "epochs":10,
        "verbose":False,
        "eval_every_items":2048,
        "eval_every_epochs":1,
        "X":X,
        "Y":Y,
        "idx":idx,
        "repetitions":n_splits,
        "seed":12345
    }

    for T in [1,2,5]:
        models.append (
            {
                "model":JaxModel,
                "step_size":1e-1,
                "n_trees":T,
                "temp_scaling":1.0,
                **shared_cfg
            }
        )

        models.append (
            {
                "model":JaxModel,
                "step_size":1e-1,
                "n_trees":T,
                "temp_scaling":2.0,
                **shared_cfg
            }
        )

for T in [16]:
    models.append(
        {
            "model":RiverModel,
            "river_model":river.ensemble.SRPClassifier(
                n_models = T,
                model = river.tree.ExtremelyFastDecisionTreeClassifier(
                    grace_period = 300,
                    split_confidence = 1e-6,
                    min_samples_reevaluate = 300,
                    leaf_prediction = "mc"
                ),
            ),
            "loss":"cross-entropy",
            "batch_size":128,
            "verbose":False,
            "epochs":10,
            "eval_every_items":2048,
            "eval_every_epochs":1,
            "X":X,
            "Y":Y,
            "idx":idx,
            "repetitions":n_splits,
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
            "batch_size":128,
            "verbose":False,
            "epochs":10,
            "eval_every_items":2048,
            "eval_every_epochs":1,
            "X":X,
            "Y":Y,
            "idx":idx,
            "repetitions":n_splits,
            "seed":12345
        }
    )

for T in [16, 32, 64, 128]:
    for bs in [32, 128, 512, 1024]:
        models.append(
            {
                "model":RandomForestClassifier,
                "bootstrap":True,
                "max_samples":bs,
                "max_depth":None,
                "loss":"cross-entropy",
                "n_estimators":T,
                "verbose":False,
                "X":X,
                "Y":Y,
                "idx":idx,
                "repetitions":n_splits,
                "random_state":12345
            }
        )

        models.append(
            {
                "model":ExtraTreesClassifier,
                "bootstrap":True,
                "max_samples":bs,
                "max_depth":None,
                "loss":"cross-entropy",
                "n_estimators":T,
                "verbose":False,
                "X":X,
                "Y":Y,
                "idx":idx,
                "repetitions":n_splits,
                "random_state":12345
            }
        )

        models.append(
            {
                "model":GradientBoostingClassifier,
                "n_estimators":T,
                "max_depth":None,
                "loss":"deviance",
                "eval_loss":"cross-entropy",
                "subsample":bs / min([len(i[0]) for i in idx]),
                "verbose":False,
                "X":X,
                "Y":Y,
                "idx":idx,
                "repetitions":n_splits,
                "random_state":12345
            }
        )

random.shuffle(models)
'''

run_experiments(basecfg, models)
