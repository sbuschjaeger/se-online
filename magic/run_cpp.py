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

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from experiment_runner.experiment_runner_v2 import run_experiments, get_ctor_arguments
from PyBPE import BiasedProxEnsemble

def cross_entropy(pred, target, epsilon=1e-12):
    #pred = np.clip(pred, epsilon, 1.0 - epsilon)
    p = softmax(pred, axis=1)
    log_likelihood = -target*np.log(p)

    return log_likelihood

def pre(cfg):
    model_ctor = cfg.pop("model")

    alpha = cfg["optimizer_cfg"].get("alpha", 1e-2)
    l_reg = cfg["optimizer_cfg"].get("lambda", 1e-3)
    n_classes = cfg["n_classes"]
    max_depth = cfg.get("max_depth", 10)
    seed = cfg.get("seed", 1234)
    model = model_ctor(max_depth, n_classes, seed, alpha, l_reg)
    return model

def post(cfg, model):
    i = cfg["run_id"]
    itrain, itest = cfg["idx"][i]
    scores = {}
    X = cfg["X"]
    Y = cfg["Y"]

    proba = np.array( model.predict_proba(X[itest]) )

    target_one_hot = np.array([ [1 if t == c else 0 for c in model.classes_] for t in Y[itest]])
    scores["test_accuracy"] = accuracy_score(Y[itest], proba.argmax(axis=1))*100.0
    scores["test_loss"] = cross_entropy(proba, target_one_hot).sum(axis=1).mean()

    if isinstance(model, BiasedProxEnsemble):
        scores["n_estimators"] = model.num_trees()
    else:
        scores["n_estimators"] = len(model.estimators_) 

    return scores

def create_mini_batches(inputs, targets, batch_size, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        start_idx += batch_size
        yield inputs[excerpt], targets[excerpt]

def fit(cfg, model):
    i = cfg["run_id"]
    itrain, itest = cfg["idx"][i]
    X = np.array(cfg["X"][itrain])
    Y = np.array(cfg["Y"][itrain])

    epochs = cfg["optimizer_cfg"].get("epochs", 50)
    batch_size = cfg["optimizer_cfg"].get("batch_size", 128)
    verbose = cfg["optimizer_cfg"].get("verbose", True)
    for epoch in range(epochs):
        mini_batches = create_mini_batches(X, Y, batch_size, True) 
        epoch_loss = 0
        batch_cnt = 0
        avg_accuarcy = 0
        epoch_nonzero = 0
        with tqdm(total=X.shape[0], ncols=135, disable = not verbose) as pbar:
            for batch in mini_batches:  
                data, target = batch 
                lsum = model.next(data, target)
                output = np.array( model.predict_proba(data) ) 
                
                epoch_loss += lsum / data.shape[0]
                epoch_nonzero += model.num_trees()
                accuracy = accuracy_score(target, output.argmax(axis=1))*100.0
                #accuracy = 0
                avg_accuarcy += accuracy
                batch_cnt += 1

                pbar.update(data.shape[0])
                desc = '[{}/{}] loss {:2.4f} acc {:2.4f} nonzero {:2.4f}'.format(
                    epoch, 
                    epochs-1, 
                    epoch_loss/batch_cnt, 
                    avg_accuarcy/batch_cnt,
                    epoch_nonzero/batch_cnt
                )
                pbar.set_description(desc)

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
        #"loss_function":cross_entropy,
        #"loss_function_deriv":cross_entropy_deriv,
        "batch_size":256,
        "alpha":1e-2,
        "lambda":l, # This depends on the strength of the base learner as well as the step size alpha
        "epochs":100
    }

    models.append(
        {
            "model":BiasedProxEnsemble,
            "n_classes":2,
            "max_depth":10,
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
