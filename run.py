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
from WindowedTree import WindowedTree

from experiment_runner.experiment_runner import run_experiments, Variation, generate_configs

# from BiasedProxEnsemble import BiasedProxEnsemble
# from SGDEnsemble import SGDEnsemble
from RiverModel import RiverModel
from PrimeModel import PrimeModel
from MoaModel import MoaModel

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
parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default=["gas-sensor"], nargs='+')
parser.add_argument("-c", "--n_configs", help="Number of configs per base learner",type=int, default=1)
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
    
    # nominal_attributes = []

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
        is_nominal = (df.nunique() == 2).values
        X = df.values.astype(np.float64)
    elif dataset == "gas-sensor":
        dfs = []
        for i in range(1,11):
            dfs.append( pd.read_csv(os.path.join("gas-sensor", "Dataset", "batch{}.dat".format(i)), header=None, delimiter = " ") )
        df = pd.concat(dfs, axis=0, ignore_index=True)

        Y = df[0].values.astype(np.int32) - 1
        df = df.drop([0], axis=1)
        is_nominal = (df.nunique() == 2).values

        X = df.values.astype(np.float64)
    else:
        exit(1)

    # TODO This might not be known in a real streaming setting. How important is it?
    # Currently the c backend assumes normalized data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    from collections import Counter
    print("Data: ", X.shape)
    print("Labels: ", Y.shape, " ", Counter(Y))
    # print("Data: ", X.shape)
    # print("Labels: ", Y.shape, " ", set(Y))
    # print("")
    # continue
    # is_nominal = np.unique(X, axis=0, return_counts=True)[1] <= 2
    # print(np.unique(X, axis=0, return_counts=True))
    nominal_attributes = ["att_" + str(j) for j, nom in enumerate(is_nominal) if nom ]
    # if len(nominal_attributes) == 0:
    #     nominal_attributes = []
    #     is_nominal = []

    experiment_cfg = {
        "X":X,
        "Y":Y,
        "verbose":True,
        "dataset":dataset,
        "seed":0
    }

    online_learner_cfg = {
        "seed":0,
        "eval_loss":"mse",
        "out_path":".",
        "verbose":args.n_jobs == 1,
        "shuffle":False,
    }

    np.random.seed(experiment_cfg["seed"])
    print("Generating random hyperparameter configurations")

    '''
    models.extend(
        generate_configs(
            {
                "model":WindowedTree,
                "model_params": {
                    "max_depth":Variation([2, 3, 4, 5, 6, 7]),
                    "seed":experiment_cfg["seed"],
                    "batch_size":Variation([2, 4, 8, 16, 32, 64, 128]),
                    "splitter" : Variation(["best", "random"]),
                    "criterion" : Variation(["gini","entropy"]), 
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
                    "max_depth":Variation([2,4,6,8]),
                    "loss":Variation(["mse","cross-entropy"]),
                    "ensemble_regularizer":"hard-L0",
                    "l_ensemble_reg":Variation([4,8,16]),
                    "tree_regularizer":None,
                    "l_tree_reg":0,
                    "normalize_weights":True,
                    "init_weight":"average",
                    "update_leaves":Variation([True, False]),
                    "seed":experiment_cfg["seed"],
                    "batch_size":Variation([4,8,32,64]),
                    "step_size":Variation(["adaptive",4e-1,3.5e-1,3e-1,1e-1,2e-1,2.5e-1,5e-1]),
                    "additional_tree_options" : {
                        "splitter" : Variation(["random","best"]),
                        "criterion" : Variation(["gini", "entropy"])
                    },
                    "backend" : "python",
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
                "model":MoaModel,
                "model_params": {
                    "moa_model":"moa.classifiers.meta.OzaBag",
                    "moa_params": {
                        "s" : Variation(["5"]),
                        "l": {
                            MoaModel.MOA_EMPTY_PLACEHOLDER : "moa.classifiers.trees.HoeffdingTree",
                            "l":"MC"
                        }
                    },
                    "nominal_attributes":nominal_attributes,
                    "moa_jar":os.path.join("moa-release-2020.12.0", "lib", "moa.jar"),
                    **online_learner_cfg
                },
                **experiment_cfg
            }, 
            n_configs=args.n_configs
        )
    )
    '''
    # moa.classifiers.meta.StreamingRandomPatches -l (  moa.classifiers.HoeffdingTree -g 50 -c 0.01 ) -s 100 --o (Percentage (M * (m / 100)))
    # -l (meta.StreamingRandomPatches -l (trees.HoeffdingTree -g 50 -c 0.01) -s 100 -o (Percentage (M * (m / 100)))
    # models.extend(
    #     generate_configs(
    #         {
    #             "model":MoaModel,
    #             "model_params": {
    #                 "moa_model":"moa.classifiers.meta.StreamingRandomPatches",
    #                 "moa_params": {
    #                     "l" : {
    #                         MoaModel.MOA_EMPTY_PLACEHOLDER:"moa.classifiers.trees.HoeffdingTree",
    #                         "g":50,
    #                         "c":0.01
    #                     },
    #                     "s" : 100,
    #                     "o" : "(Percentage (M * (m / 100)))"
    #                 },
    #                 "nominal_attributes":nominal_attributes,
    #                 "moa_jar":os.path.join("moa-release-2020.12.0", "lib", "moa.jar"),
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
    #             "model":MoaModel,
    #             "model_params": {
    #                 "moa_model":"moa.classifiers.trees.EFDT",
    #                 "moa_params": {
    #                     "l" : Variation(["MC"]),
    #                     "c" : Variation([0.1]),
    #                     "g" : Variation([50])
    #                 },
    #                 "nominal_attributes":nominal_attributes,
    #                 "moa_jar":os.path.join("moa-release-2020.12.0", "lib", "moa.jar"),
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
    #             "model":MoaModel,
    #             "model_params": {
    #                 #"moa_model":"HoeffdingTreeClassifier",
    #                 "moa_model":"ExtremelyFastDecisionTreeClassifier",
    #                 "moa_params": {
    #                     "gracePeriodOption" : Variation([50]),
    #                     "leafpredictionOption" : "MC",
    #                     "splitConfidenceOption": Variation([0.1]),
    #                 },
    #                 "nominal_attributes":nominal_attributes,
    #                 "moa_jar":os.path.join("moa-release-2020.12.0", "lib", "moa.jar"),
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         }, 
    #         n_configs=args.n_configs
    #     )
    # )

    models.extend(
        generate_configs(
            {
                "model":RiverModel,
                "model_params": {
                    "river_model":"HoeffdingTreeClassifier",
                    "river_params": {
                        "grace_period" : Variation([50]),
                        "split_confidence" : Variation([0.1]),
                        "leaf_prediction" : Variation(["mc"]),
                        "nominal_attributes":nominal_attributes
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
    #             "model":RiverModel,
    #             "model_params": {
    #                 "river_model": "SRP",
    #                 "river_params": {
    #                     "model":"HoeffdingTreeClassifier",
    #                     "n_models":32,
    #                     "model_params": {
    #                         "grace_period" : 50,
    #                         "split_confidence" : 0.1,
    #                         "leaf_prediction" : "nba",
    #                         "nominal_attributes":nominal_attributes
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
    #             "model":PrimeModel,
    #             "model_params": {
    #                 "max_depth":8,
    #                 "loss":"cross-entropy",
    #                 "ensemble_regularizer":"hard-L1",
    #                 "l_ensemble_reg":32,
    #                 "tree_regularizer":None,
    #                 "l_tree_reg":0,
    #                 "normalize_weights":True,
    #                 "init_weight":"average",
    #                 "update_leaves":True,
    #                 "seed":experiment_cfg["seed"],
    #                 "batch_size":8,
    #                 "step_size":1e-1,
    #                 "additional_tree_options" : {
    #                     "tree_init_mode" : "train",
    #                     "is_nominal":is_nominal
    #                 },
    #                 "backend" : "c++",
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         }, 
    #         n_configs=args.n_configs
    #     )
    # )

    # 96.39 % Accuarcy
    # Maybe average is the problem?
    # models.extend(
    #     generate_configs(
    #         {
    #             "model":PrimeModel,
    #             "model_params": {
    #                 "max_depth":Variation([2, 3, 4, 5, 6, 7]),
    #                 "loss":Variation(["mse"]),
    #                 "ensemble_regularizer":"hard-L1",
    #                 "l_ensemble_reg":Variation([16,32,64,128]),
    #                 "tree_regularizer":None,
    #                 "l_tree_reg":0,
    #                 "normalize_weights":True,
    #                 "init_weight":"average",
    #                 "update_leaves":Variation([True]), #True, 
    #                 "seed":experiment_cfg["seed"],
    #                 "batch_size":Variation([4, 8, 16, 32, 64, 128]),
    #                 "step_size":Variation([1, 5e-1, 2e-1, 1e-1, 1e-2]), #1e-1,5e-1,1,2,3,5,7,
    #                 "additional_tree_options" : {
    #                     "tree_init_mode" : Variation(["train"]),
    #                     "is_nominal":is_nominal
    #                 },
    #                 "backend" : "c++",
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
    #             "model":PrimeModel,
    #             "model_params": {
    #                 "max_depth":Variation([2, 3, 4, 5]),
    #                 "loss":Variation(["mse"]),
    #                 "ensemble_regularizer":"hard-L1",
    #                 "l_ensemble_reg":Variation([16,32,64,128]),
    #                 "tree_regularizer":None,
    #                 "l_tree_reg":0,
    #                 "normalize_weights":True,
    #                 "init_weight":"average",
    #                 "update_leaves":Variation([True]), #True, 
    #                 "seed":experiment_cfg["seed"],
    #                 "batch_size":Variation([4, 8, 16, 32, 64, 128]),
    #                 "step_size":Variation([1, 5e-1, 2e-1, 1e-1, 1e-2]), #1e-1,5e-1,1,2,3,5,7,
    #                 "additional_tree_options" : {
    #                     "splitter" : Variation(["best"]),
    #                     "criterion" : "gini"
    #                 },
    #                 "backend" : "python",
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
    #             "model":PrimeModel,
    #             "model_params": {
    #                 "max_depth":Variation([3]),
    #                 "loss":Variation(["mse"]),
    #                 "ensemble_regularizer":"hard-L1",
    #                 "l_ensemble_reg":Variation([16]),
    #                 "tree_regularizer":None,
    #                 "l_tree_reg":0,
    #                 "normalize_weights":True,
    #                 "init_weight":"average",
    #                 "update_leaves":Variation([True]),
    #                 "seed":experiment_cfg["seed"],
    #                 "batch_size":Variation([32]),
    #                 "step_size":Variation([1e-1]), #1e-1,5e-1,1,2,3,5,7,
    #                 "additional_tree_options" : {
    #                     "splitter" : "best",
    #                     "criterion" : "gini"
    #                 },
    #                 "backend" : "python",
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
    #             "model":PrimeModel,
    #             "model_params": {
    #                 "max_depth":Variation([5]),
    #                 "loss":Variation(["mse"]),
    #                 "ensemble_regularizer":"hard-L1",
    #                 "l_ensemble_reg":Variation([4]),
    #                 "tree_regularizer":None,
    #                 "l_tree_reg":0,
    #                 "normalize_weights":False,
    #                 #"init_weight":"average",
    #                 "init_weight":0.1,
    #                 "update_leaves":Variation([True]),
    #                 "seed":experiment_cfg["seed"],
    #                 "batch_size":Variation([32]),
    #                 "step_size":Variation([1e-1]),
    #                 "additional_tree_options" : {
    #                     "splitter" : Variation(["best"]), 
    #                     "criterion" : "gini"
    #                 },
    #                 "backend" : "python",
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
    #             "model":PrimeModel,
    #             "model_params": {
    #                 "max_depth":Variation([2,3,4,5,6,7,8,9,10]),
    #                 "loss":Variation(["cross-entropy","mse"]),
    #                 "ensemble_regularizer":"hard-L1",
    #                 "l_ensemble_reg":Variation([16,32,64,128,256,512,1024]),
    #                 "tree_regularizer":None,
    #                 "l_tree_reg":0,
    #                 "normalize_weights":True,
    #                 "init_weight":"average",
    #                 "update_leaves":Variation([True, False]),
    #                 "seed":experiment_cfg["seed"],
    #                 "batch_size":Variation([8,32,128,512,1024]),
    #                 "step_size":Variation([1e-1,1e-2,1e-3]), #1e-1,5e-1,1,2,3,5,7,
    #                 "additional_tree_options" : {
    #                     "tree_init_mode" : Variation(["train", "fully-random", "random"]),
    #                     "tree_init_mode" : Variation(["train", "fully-random", "random"]),
    #                     "is_nominal":is_nominal
    #                 },
    #                 "backend" : "c++",
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         }, 
    #         n_configs=args.n_configs
    #     )
    # )
    
    '''
    models.extend(
        generate_configs(
            {
                "model":RiverModel,
                "model_params": {
                    "river_model": "HoeffdingTreeClassifier",
                    "river_params": {
                        "grace_period" : Variation([10,50,100,200,500]),
                        "split_confidence" : Variation([0.1, 0.01, 0.001]),
                        "leaf_prediction" : Variation(["mc", "nba"]),
                        "nominal_attributes":nominal_attributes
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
                "model":RiverModel,
                "model_params": {
                    "river_model": "ExtremelyFastDecisionTreeClassifier",
                    "river_params": {
                        "grace_period" : Variation([10,50,100,200,500]),
                        "split_confidence" : Variation([0.1, 0.01, 0.001]),
                        "leaf_prediction" : Variation(["mc", "nba"]),
                        "nominal_attributes":nominal_attributes
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
                "model":RiverModel,
                "model_params": {
                    "river_model": "SRP",
                    "river_params": {
                        "model":"HoeffdingTreeClassifier",
                        "n_models":Variation([2,4,8,16,32]),
                        "model_params": {
                            "grace_period" : Variation([10,50,100,200,500]),
                            "split_confidence" : Variation([0.1, 0.01, 0.001]),
                            "leaf_prediction" : Variation(["mc", "nba"]),
                            "nominal_attributes":nominal_attributes
                        }
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
                "model":RiverModel,
                "model_params": {
                    "river_model": "SRP",
                    "river_params": {
                        "model":"ExtremelyFastDecisionTreeClassifier",
                        "n_models":Variation([2,4,8,16,32]),
                        "model_params": {
                            "grace_period" : Variation([10,50,100,200,500]),
                            "split_confidence" : Variation([0.1, 0.01, 0.001]),
                            "leaf_prediction" : Variation(["mc", "nba"]),
                            "nominal_attributes":nominal_attributes
                        }
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
                "model":RiverModel,
                "model_params": {
                    "river_model": "AdaBoostClassifier",
                    "river_params": {
                        "model":"HoeffdingTreeClassifier",
                        "n_models":Variation([2,4,8,16,32]),
                        "model_params": {
                            "grace_period" : Variation([10,50,100,200,500]),
                            "split_confidence" : Variation([0.1, 0.01, 0.001]),
                            "leaf_prediction" : Variation(["mc", "nba"]),
                            "nominal_attributes":nominal_attributes
                        }
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
                "model":RiverModel,
                "model_params": {
                    "river_model": "AdaBoostClassifier",
                    "river_params": {
                        "model":"ExtremelyFastDecisionTreeClassifier",
                        "n_models":Variation([2,4,8,16,32]),
                        "model_params": {
                            "grace_period" : Variation([10,50,100,200,500]),
                            "split_confidence" : Variation([0.1, 0.01, 0.001]),
                            "leaf_prediction" : Variation(["mc", "nba"]),
                            "nominal_attributes":nominal_attributes
                        }
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
                "model":RiverModel,
                "model_params": {
                    "river_model": "BaggingClassifier",
                    "river_params": {
                        "model":"HoeffdingTreeClassifier",
                        "n_models":Variation([2,4,8,16,32]),
                        "model_params": {
                            "grace_period" : Variation([10,50,100,200,500]),
                            "split_confidence" : Variation([0.1, 0.01, 0.001]),
                            "leaf_prediction" : Variation(["mc", "nba"]),
                            "nominal_attributes":nominal_attributes
                        }
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
                "model":RiverModel,
                "model_params": {
                    "river_model": "BaggingClassifier",
                    "river_params": {
                        "model":"ExtremelyFastDecisionTreeClassifier",
                        "n_models":Variation([2,4,8,16,32]),
                        "model_params": {
                            "grace_period" : Variation([10,50,100,200,500]),
                            "split_confidence" : Variation([0.1, 0.01, 0.001]),
                            "leaf_prediction" : Variation(["mc", "nba"]),
                            "nominal_attributes":nominal_attributes
                        }
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
                "model":RiverModel,
                "model_params": {
                    "river_model": "AdaptiveRandomForestClassifier",
                    "river_params": {
                        "grace_period" : Variation([10,50,100,200,500]),
                        "split_confidence" : Variation([0.1, 0.01, 0.001]),
                        "leaf_prediction" : Variation(["mc", "nba"]),
                        "n_models":Variation([2,4,8,16,32]),
                        "max_features":Variation([0.25,0.5,0.75]),
                        "nominal_attributes":nominal_attributes
                    },
                    **online_learner_cfg
                },
                **experiment_cfg
            },
            n_configs=args.n_configs
        )
    )
    '''

    # models.extend(
    #     generate_configs(
    #         {
    #             "model":JaxModel,
    #             "model_params": {
    #                 "loss":Variation(["cross-entropy","mse"]),
    #                 "step_size":Variation([1e-3,1e-2,1e-1,5e-1,1,2]),
    #                 "max_depth":Variation([2,3,4,5]),
    #                 "n_trees":Variation([1,2,4]),
    #                 "batch_size":Variation([8,16,32,64,128,256]),
    #                 **online_learner_cfg
    #             },
    #             **experiment_cfg
    #         },
    #         n_configs=args.n_configs
    #     )
    # )


# random.shuffle(models)


run_experiments(basecfg, models)
