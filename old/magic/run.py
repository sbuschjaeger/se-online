#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from datetime import datetime
from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from experiment_runner.experiment_runner_v2 import run_experiments, get_ctor_arguments

sys.path.append("../")
from BiasedProxEnsemble import BiasedProxEnsemble

def cross_entropy(pred, target, epsilon=1e-12):
    #pred = np.clip(pred, epsilon, 1.0 - epsilon)
    p = softmax(pred, axis=1)
    log_likelihood = -target*np.log(p)

    return log_likelihood

def cross_entropy_deriv(pred, target):
    m = target.shape[0]
    grad = softmax(pred, axis=1)
    grad[range(m),target.argmax(axis=1)] -= 1
    #grad = grad/m
    return grad

def pre(cfg):
    model_ctor = cfg.pop("model")
    tmpcfg = cfg
    expected = {}
    for key in get_ctor_arguments(model_ctor):
        if key in tmpcfg:
            expected[key] = tmpcfg[key]
    
    model = model_ctor(**expected)
    return model

def post(cfg, model):
    i = cfg["run_id"]
    itrain, itest = cfg["idx"][i]
    scores = {}
    X = cfg["X"]
    Y = cfg["Y"]

    if isinstance(model, BiasedProxEnsemble):
        proba, _ = model.predict_proba(X[itest])
    else:
        proba = model.predict_proba(X[itest])

    target_one_hot = np.array([ [1 if t == c else 0 for c in model.classes_] for t in Y[itest]])
    scores["test_accuracy"] = accuracy_score(Y[itest], proba.argmax(axis=1))*100.0
    scores["test_loss"] = cross_entropy(proba, target_one_hot).sum(axis=1).mean()

    if hasattr(model, "estimator_weights_"):
        scores["n_estimators"] = np.count_nonzero(model.estimator_weights_)
    else:
        scores["n_estimators"] = len(model.estimators_) 

    return scores

def fit(cfg, model):
    i = cfg["run_id"]
    itrain, itest = cfg["idx"][i]
    model.fit(cfg["X"][itrain],cfg["Y"][itrain])
    return model

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local", help="Run on local machine",action="store_true", default=False)
parser.add_argument("-r", "--ray", help="Run via Ray",action="store_true", default=False)
parser.add_argument("--ray_head", help="Run via Ray",action="store_true", default="auto")
parser.add_argument("--redis_password", help="Run via Ray",action="store_true", default="5241590000000000")
args = parser.parse_args()

if (args.local and args.ray) or (not args.local and not args.ray):
    print("Either you specified to use both, ray _and_ local mode or you specified to use none of both. Please choose either. Defaulting to `local` processing.")
    args.local = True

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

# for T in [128, 256, 512, 1024, 2048]:
#     models.append(
#         {
#             "model":RandomForestClassifier,
#             "n_estimators":T,
#             "verbose":False,
#             "X":X,
#             "Y":Y,
#             "idx":idx,
#             "repetitions":n_splits
#         }
#     )

#     models.append(
#         {
#             "model":ExtraTreesClassifier,
#             "n_estimators":T,
#             "verbose":False,
#             "X":X,
#             "Y":Y,
#             "idx":idx,
#             "repetitions":n_splits
#         }
#     )

#     models.append(
#         {
#             "model":AdaBoostClassifier,
#             "algorithm" : "SAMME.R",
#             "n_estimators":T,
#             "verbose":False,
#             "X":X,
#             "Y":Y,
#             "idx":idx,
#             "repetitions":n_splits
#         }
#     )

for l in [ 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
    optimizer_cfg = {
        "loss_function":cross_entropy,
        "loss_function_deriv":cross_entropy_deriv,
        "batch_size":256,
        "alpha":1e-2,
        "lambda":l, # This depends on the strength of the base learner as well as the step size alpha
        "epochs":100
    }

    models.append(
        {
            "model":BiasedProxEnsemble,
            "optimizer_cfg":optimizer_cfg,
            "verbose":False,
            "X":X,
            "Y":Y,
            "idx":idx,
            "max_depth":20,
            "mode":"sklearn-random",
            "repetitions":n_splits
        }
    )

run_experiments(basecfg, models)
