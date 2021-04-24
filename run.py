#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import random
from scipy.io.arff import loadarff
import copy

from functools import partial

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.metrics import roc_auc_score

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from experiment_runner.experiment_runner import run_experiments, Variation, generate_configs

import river
from river import tree
from river.ensemble import AdaptiveRandomForestClassifier

from JaxModel import JaxModel

from experiment_runner.experiment_runner import run_experiments, Variation, generate_configs

# from BiasedProxEnsemble import BiasedProxEnsemble
# from SGDEnsemble import SGDEnsemble
from RiverModel import RiverModel
from PrimeModel import PrimeModel

def pre(cfg):
    model_ctor = cfg.pop("model")
    if model_ctor == RiverModel:
        base_model = cfg["model_params"].pop("river_model")
        river_params = cfg["model_params"].pop("river_params")

        if base_model == "HoeffdingTreeClassifier":
            base_model = river.tree.HoeffdingTreeClassifier(**river_params)
        elif base_model == "ExtremelyFastDecisionTreeClassifier":
            base_model = river.tree.ExtremelyFastDecisionTreeClassifier(**river_params)
        elif base_model == "AdaptiveRandomForestClassifier":
            base_model = river.ensemble.AdaptiveRandomForestClassifier(**river_params)
        elif base_model in ["SRP", "BaggingClassifier", "AdaBoostClassifier"]:
            river_base = river_params.pop("model")
            if river_base == "HoeffdingTreeClassifier":
                river_base = river.tree.HoeffdingTreeClassifier(**river_params.pop("model_params"))
            elif river_base == "ExtremelyFastDecisionTreeClassifier":
                river_base = river.tree.ExtremelyFastDecisionTreeClassifier(**river_params.pop("model_params"))

            if base_model == "SRP":
                base_model = river.ensemble.SRPClassifier(model = river_base, **river_params)
            elif base_model == "BaggingClassifier":
                base_model = river.ensemble.BaggingClassifier(model = river_base, **river_params)
            elif base_model == "AdaBoostClassifier":
                base_model = river.ensemble.AdaBoostClassifier(model = river_base, **river_params)

        cfg["model_params"]["river_model"] = base_model

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


parser = argparse.ArgumentParser()
parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default=["elec"], nargs='+')
parser.add_argument("-c", "--n_configs", help="Number of configs per base learner",type=int, default=50)
parser.add_argument("-t", "--timeout", help="Maximum number of seconds per run. If the runtime exceeds the provided value, stop execution",type=int, default=7200)
args = parser.parse_args()

# TODO Add algos as parameter?
# ARF
# Bag+ET
# Bag+HT
# AB+ET
# AB+HT
# SRP+ET
# SRP+HT
# ET
# HT
# SDT
# Prime

if len(args.dataset) == 1:
    outpath = args.dataset[0]
    args.dataset = args.dataset
else:
    outpath = "multi"

if args.n_jobs == 1:
    basecfg = {
        "out_path":os.path.join(outpath, "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "single",
        "verbose":True,
        "timeout":args.timeout
    }
else:
    basecfg = {
        "out_path":os.path.join(outpath, "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S')),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "multiprocessing",
        "num_cpus":args.n_jobs,
        "verbose":True,
        "timeout":args.timeout
    }

models = []
for dataset in args.dataset:
    print("Loading {}".format(dataset))

    # "eeg", "elec", "nomao"
    if dataset in ["elec"]:
        #if dataset == "eeg":
        #    data, meta = loadarff(os.path.join("eeg", "EEG Eye State.arff"))
        if dataset == "elec":
            data, meta = loadarff(os.path.join("elec", "elecNormNew.arff"))
        #else:
        #    data, meta = loadarff(os.path.join("nomao", "nomao.arff.txt"))

        Xdict = {}
        for cname, ctype in zip(meta.names(), meta.types()):
            # Get the label attribute for the specific dataset:
            #   eeg: eyeDetection
            #   elec: class
            #   nomao: Class
            if cname in ["eyeDetection", "class",  "Class"]:
                enc = LabelEncoder()
                Xdict["label"] = enc.fit_transform(data[cname])
            else:
                Xdict[cname] = data[cname]
        df = pd.DataFrame(Xdict)
        df = pd.get_dummies(df)
        Y = df["label"].values.astype(np.int32)
        df = df.drop("label", axis=1)

        X = df.values.astype(np.float64)
    elif dataset == "gas-sensor":
        dfs = []
        for i in range(1,11):
            dfs.append( pd.read_csv(os.path.join("gas-sensor", "Dataset", "batch{}.dat".format(i)), header=None, delimiter = " ") )
        df = pd.concat(dfs, axis=0, ignore_index=True)

        Y = df[0].values.astype(np.int32) - 1
        df = df.drop([0], axis=1)
        X = df.values.astype(np.float64)
    else:
        exit(1)

    # TODO This might not be known in a real streaming setting. How important is it?
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    from collections import Counter
    print("Data: ", X.shape, " ", X[0:2,:])
    print("Labels: ", Y.shape, " ", Counter(Y))
    # print("Data: ", X.shape)
    # print("Labels: ", Y.shape, " ", set(Y))
    # print("")
    # continue

    experiment_cfg = {
        "X":X,
        "Y":Y,
        "verbose":True,
        "eval_loss":"cross-entropy",
        "seed":0
    }

    online_learner_cfg = {
        "epochs":1,
        "out_path":".",
        "sliding_window":True,
        "verbose":args.n_jobs == 1,
        "shuffle":False,
        "eval_every_epochs":1   
    }

    np.random.seed(experiment_cfg["seed"])

    models = []
    print("Generating random hyperparameter configurations")

    models.extend(
        generate_configs(
            {
                "model":PrimeModel,
                "model_params": {
                    "max_depth":Variation([2,3,4,5,6,7,8,9,10]),
                    "loss":Variation(["cross-entropy","mse"]),
                    "ensemble_regularizer":"hard-L1",
                    "l_ensemble_reg":Variation([16,32,64,128,256,512]),
                    "tree_regularizer":None,
                    "l_tree_reg":0,
                    "normalize_weights":True,
                    "init_weight":"average",
                    "update_leaves":Variation([True, False]),
                    "seed":experiment_cfg["seed"],
                    "batch_size":Variation([4,8,16,32,64,128,256]),
                    "step_size":Variation([10,12,15,20]), #1e-1,5e-1,1,2,3,5,7,
                    "additional_tree_options" : {
                        "splitter" : "best", "criterion" : "gini"
                    },
                    **online_learner_cfg
                },
                **experiment_cfg
            }, 
            n_configs=args.n_configs
        )
    )
    
    models.extend(
        generate_configs(
            {
                "model":PrimeModel,
                "model_params": {
                    "max_depth":Variation([2,3,4,5,6,7,8,9,10]),
                    "loss":Variation(["cross-entropy","mse"]),
                    "ensemble_regularizer":"hard-L1",
                    "l_ensemble_reg":Variation([16,32,64,128,256,512]),
                    "tree_regularizer":None,
                    "l_tree_reg":0,
                    "normalize_weights":True,
                    "init_weight":"average",
                    "update_leaves":Variation([True, False]),
                    "seed":experiment_cfg["seed"],
                    "batch_size":Variation([4,8,16,32,64,128,256]),
                    "step_size":Variation([10,12,15,20]), #1e-1,5e-1,1,2,3,5,7,
                    "additional_tree_options" : {
                        "splitter" : "random", "criterion" : "gini"
                    },
                    **online_learner_cfg
                },
                **experiment_cfg
            }, 
            n_configs=args.n_configs
        )
    )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":JaxModel,
    #             "model_params": {
    #                 "loss":"cross-entropy",
    #                 "step_size":Variation([1e-3,1e-2,1e-1,5e-1,1,2]),
    #                 "max_depth":Variation([2,3,4,5]),
    #                 "n_trees":Variation([1,2,4,8]),
    #                 "batch_size":Variation([8,16,32,64,128,256]),
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "HoeffdingTreeClassifier",
    #                 "river_params": {
    #                     "grace_period" : Variation([10,50,100,200,500]),
    #                     "split_confidence" : Variation([0.1, 0.01, 0.001]),
    #                     "leaf_prediction" : Variation(["mc", "nba"])
    #                 },
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "ExtremelyFastDecisionTreeClassifier",
    #                 "river_params": {
    #                     "grace_period" : Variation([10,50,100,200,500]),
    #                     "split_confidence" : Variation([0.1, 0.01, 0.001]),
    #                     "leaf_prediction" : Variation(["mc", "nba"])
    #                 },
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "SRP",
    #                 "river_params": {
    #                     "model":"HoeffdingTreeClassifier",
    #                     "n_models":Variation([2,4,8,16,32]),
    #                     "model_params": {
    #                         "grace_period" : Variation([10,50,100,200,500]),
    #                         "split_confidence" : Variation([0.1, 0.01, 0.001]),
    #                         "leaf_prediction" : Variation(["mc", "nba"])
    #                     }
    #                 },
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "SRP",
    #                 "river_params": {
    #                     "model":"ExtremelyFastDecisionTreeClassifier",
    #                     "n_models":Variation([2,4,8,16,32]),
    #                     "model_params": {
    #                         "grace_period" : Variation([10,50,100,200,500]),
    #                         "split_confidence" : Variation([0.1, 0.01, 0.001]),
    #                         "leaf_prediction" : Variation(["mc", "nba"])
    #                     }
    #                 },
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "AdaBoostClassifier",
    #                 "river_params": {
    #                     "model":"HoeffdingTreeClassifier",
    #                     "n_models":Variation([2,4,8,16,32]),
    #                     "model_params": {
    #                         "grace_period" : Variation([10,50,100,200,500]),
    #                         "split_confidence" : Variation([0.1, 0.01, 0.001]),
    #                         "leaf_prediction" : Variation(["mc", "nba"])
    #                     }
    #                 },
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "AdaBoostClassifier",
    #                 "river_params": {
    #                     "model":"ExtremelyFastDecisionTreeClassifier",
    #                     "n_models":Variation([2,4,8,16,32]),
    #                     "model_params": {
    #                         "grace_period" : Variation([10,50,100,200,500]),
    #                         "split_confidence" : Variation([0.1, 0.01, 0.001]),
    #                         "leaf_prediction" : Variation(["mc", "nba"])
    #                     }
    #                 },
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "BaggingClassifier",
    #                 "river_params": {
    #                     "model":"HoeffdingTreeClassifier",
    #                     "n_models":Variation([2,4,8,16,32]),
    #                     "model_params": {
    #                         "grace_period" : Variation([10,50,100,200,500]),
    #                         "split_confidence" : Variation([0.1, 0.01, 0.001]),
    #                         "leaf_prediction" : Variation(["mc", "nba"])
    #                     }
    #                 },
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "BaggingClassifier",
    #                 "river_params": {
    #                     "model":"ExtremelyFastDecisionTreeClassifier",
    #                     "n_models":Variation([2,4,8,16,32]),
    #                     "model_params": {
    #                         "grace_period" : Variation([10,50,100,200,500]),
    #                         "split_confidence" : Variation([0.1, 0.01, 0.001]),
    #                         "leaf_prediction" : Variation(["mc", "nba"])
    #                     }
    #                 },
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "AdaptiveRandomForestClassifier",
    #                 "river_params": {
    #                     "grace_period" : Variation([10,50,100,200,500]),
    #                     "split_confidence" : Variation([0.1, 0.01, 0.001]),
    #                     "leaf_prediction" : Variation(["mc", "nba"]),
    #                     "n_models":Variation([2,4,8,16,32]),
    #                     "max_features":Variation([0.25,0.5,0.75])
    #                 },
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )

random.shuffle(models)

run_experiments(basecfg, models)
