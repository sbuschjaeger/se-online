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

from TorchModel import TorchModel
from WindowedTree import WindowedTree

from experiment_runner.experiment_runner import run_experiments, Variation, generate_configs

# from BiasedProxEnsemble import BiasedProxEnsemble
# from SGDEnsemble import SGDEnsemble
from RiverModel import RiverModel
from SEModel import SEModel
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

def main(args):
    # if len(args.dataset) == 1:
    #     dataset = args.dataset[0]
    #     args.dataset = args.dataset
    # else:
    #     dataset = "multi"


    for dataset in args.dataset:
        outpath = os.path.join(args.out_path, dataset, "results", datetime.now().strftime('%d-%m-%Y-%H:%M:%S'))

        if args.n_jobs == 1:
            basecfg = {
                "out_path":outpath,
                "pre": pre,
                "post": post,
                "fit": fit,
                "backend": "single",
                "verbose":True,
                "timeout":args.timeout
            }
        else:
            basecfg = {
                "out_path":outpath,
                "pre": pre,
                "post": post,
                "fit": fit,
                "backend": "multiprocessing",
                "num_cpus":args.n_jobs,
                "verbose":True,
                "timeout":args.timeout
            }
        models = []

        print("Loading {}".format(dataset))
        # ./run.py -d airlines covtype elec gmsc gas-sensor nomao agrawal_a agrawal_g led_a led_g rbf_f rbf_m weather spam -c 50 -j 60 -o /rdata/s01b_ls8_000/buschjae
        # kdd99: Groß und sehr unbalanciert -> accuracy nah 100%
        # occupancy: kappaC -> 0
        # eeg: kappaC -> 0
        # ads: kappaC -> 0
        if dataset in ["eeg", "elec", "agrawal_a", "agrawal_g", "led_a", "led_g", "rbf_f", "rbf_m", "airlines", "covtype", "nomao", "ads", "kdd99", "spam"]:
            if dataset == "eeg":
                data, meta = loadarff(os.path.join("eeg", "EEG Eye State.arff"))
            elif dataset == "elec":
                data, meta = loadarff(os.path.join("elec", "elecNormNew.arff"))
            elif dataset == "agrawal_a":
                data, meta = loadarff(os.path.join("synthetic", "agrawal_a.arff"))
            elif dataset == "agrawal_g":
                data, meta = loadarff(os.path.join("synthetic", "agrawal_g.arff"))
            elif dataset == "led_a":
                data, meta = loadarff(os.path.join("synthetic", "led_a.arff"))
            elif dataset == "led_g":
                data, meta = loadarff(os.path.join("synthetic", "led_g.arff"))
            elif dataset == "rbf_f":
                data, meta = loadarff(os.path.join("synthetic", "rbf_f.arff"))
            elif dataset == "rbf_m":
                data, meta = loadarff(os.path.join("synthetic", "rbf_m.arff"))
            elif dataset == "airlines":
                data, meta = loadarff(os.path.join("airlines", "airlines.arff"))
            elif dataset == "covtype":
                data, meta = loadarff(os.path.join("covtype", "covtypeNorm.arff"))
            elif dataset == "nomao":
                data, meta = loadarff(os.path.join("nomao", "nomao.arff.txt"))
            elif dataset == "ads":
                data, meta = loadarff(os.path.join("ads", "internet_ads.arff"))
            elif dataset == "spam":
                data, meta = loadarff(os.path.join("spam", "spam_corpus.arff"))
            elif dataset == "kdd99":
                data, meta = loadarff(os.path.join("kdd99", "kdd99.arff"))

            Xdict = {}
            for cname, ctype in zip(meta.names(), meta.types()):
                # Get the label attribute for the specific dataset:
                #   eeg: eyeDetection
                #   elec: class
                #   nomao: Class
                #   synthetic datasets: class
                #   airlines: Delay
                #   covtype: class
                #   internet_ads: class
                #   spam: spamorlegitimate
                #   kdd99: label
                if cname in ["eyeDetection", "class",  "Class", "Delay", "spamorlegitimate", "label"]:
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
        elif dataset == "gmsc":
            df = pd.read_csv(os.path.join("gmsc","cs-training.csv"))
            df = df.dropna()
            Y = df["SeriousDlqin2yrs"].values.astype(np.int32)
            df = df.drop(["SeriousDlqin2yrs"], axis=1)
            is_nominal = (df.nunique() == 2).values
            X = df.values.astype(np.float64)
        elif dataset == "occupancy":
            df = pd.read_csv(os.path.join("occupancy","data.csv"))
            Y = df["Occupancy"].values.astype(np.int32)
            df = df.drop(["date", "Occupancy"], axis=1)
            is_nominal = (df.nunique() == 2).values
            X = df.values.astype(np.float64)
        elif dataset == "weather":
            df = pd.read_csv(os.path.join("weather","weather.csv"))
            Y = df["target"].values.astype(np.int32)
            df = df.drop(["target"], axis=1)
            is_nominal = (df.nunique() == 2).values
            X = df.values.astype(np.float64)
        else:
            exit(1)

        # scaler = MinMaxScaler()
        # X = scaler.fit_transform(X)

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

        ## DEBUG ## 
        # models.extend(
        #     generate_configs(
        #         {
        #             "model":MoaModel,
        #             "model_params": {
        #                 "moa_model":"moa.classifiers.meta.StreamingRandomPatches",
        #                 "moa_params": {
        #                     "l" : {
        #                         MoaModel.MOA_EMPTY_PLACEHOLDER: Variation(["moa.classifiers.trees.HoeffdingTree"]),
        #                         "g":Variation([50]),
        #                         "c":Variation([0.1]),
        #                         "l":Variation(["NB"])
        #                     },
        #                     "x" : Variation(["(ADWINChangeDetector -a 1.0E-1)"]), #1.0E-5
        #                     "s" : Variation([32]),
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
        #             "model":SEModel,
        #             "model_params": {
        #                 "loss":Variation(["mse"]),
        #                 "ensemble_regularizer":"hard-L0",
        #                 "l_ensemble_reg":Variation([32]),
        #                 "tree_regularizer":None,
        #                 "l_tree_reg":0,
        #                 "normalize_weights":True,
        #                 "update_leaves":Variation([True]),
        #                 "seed":experiment_cfg["seed"],
        #                 "batch_size":Variation([256]),
        #                 "step_size":None,
        #                 "l_l2_reg":None,
        #                 "rho":Variation([0.9]),
        #                 "burnin_steps":0,
        #                 "additional_tree_options" : {
        #                     "tree_init_mode" : Variation(["train"]),
        #                     #"splitter":"best",
        #                     "max_depth":Variation([5]),#sklearn / c++ (2^(d+1) - 1)*M
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
        #             "model":TorchModel,
        #             "model_params": {
        #                 "loss":Variation(["mse"]),
        #                 "step_size":Variation([1e-1]),
        #                 "max_depth":Variation([2]),
        #                 "n_trees":Variation([4]),
        #                 "batch_size":Variation([32]),
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
        #             "model":WindowedTree,
        #             "model_params": {
        #                 "additional_tree_options": {
        #                     "max_depth":Variation([10]),
        #                     "splitter" : Variation(["best"]),
        #                     "criterion" : Variation(["gini"]), 
        #                 },
        #                 "seed":experiment_cfg["seed"],
        #                 "batch_size":Variation([2**12]),
        #                 "backend":"python",
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
        #             "model":WindowedTree,
        #             "model_params": {
        #                 "additional_tree_options": {
        #                     "max_depth":Variation([10]),
        #                     "tree_init_mode" : Variation(["train"])
        #                 },
        #                 "seed":experiment_cfg["seed"],
        #                 "batch_size":Variation([2**12]),
        #                 "backend":"c++",
        #                 **online_learner_cfg
        #             },
        #             **experiment_cfg
        #         }, 
        #         n_configs=args.n_configs
        #     )
        # )

        ## DEBUG END ##
        # models.extend(
        #     generate_configs(
        #         {
        #             "model":TorchModel,
        #             "model_params": {
        #                 "loss":Variation(["cross-entropy","mse"]),
        #                 "step_size":Variation([1e-3,1e-2,1e-1,5e-1]),
        #                 "max_depth":Variation([1,2,3,4,5]),
        #                 "n_trees":Variation([4,8,16,32]),
        #                 "batch_size":Variation([2**i for i in range(4, 14)]),
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
                    "model":SEModel,
                    "model_params": {
                        "loss":Variation(["mse"]),
                        "ensemble_regularizer":"hard-L0",
                        "l_ensemble_reg":Variation([4,8,16,32,64,128,256]),
                        "tree_regularizer":None,
                        "l_tree_reg":0,
                        "l_l2_reg":None,
                        "normalize_weights":True,
                        "update_leaves":Variation([False]),
                        "seed":experiment_cfg["seed"],
                        "batch_size":Variation([2**i for i in range(4, 14)]),
                        "step_size":None,
                        "rho":Variation([0.1*i for i in range(1,10)]),
                        "burnin_steps":Variation([5,10]),
                        "additional_tree_options" : {
                            "max_depth":Variation([2,4,8,12,15]),
                            "tree_init_mode" : Variation(["train","random"]),
                            "max_features":Variation([0, int(np.sqrt(X.shape[1]))])
                        },
                        "backend" : "c++",
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
        #             "model":WindowedTree,
        #             "model_params": {
        #                 "seed":experiment_cfg["seed"],
        #                 "batch_size":Variation([2**i for i in range(4, 14)]),
        #                 "additional_tree_options": {
        #                     "max_depth":Variation([2,4,8,12,15]),
        #                     "tree_init_mode" : Variation(["train","random"]),
        #                     "max_features":Variation([0, int(np.sqrt(X.shape[1]))])
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
        #             "model":MoaModel,
        #             "model_params": {
        #                 "moa_model":"moa.classifiers.meta.StreamingRandomPatches",
        #                 "moa_params": {
        #                     "l" : {
        #                         MoaModel.MOA_EMPTY_PLACEHOLDER: Variation(["moa.classifiers.trees.HoeffdingTree","moa.classifiers.trees.EFDT"]),
        #                         "g":Variation([50,100,250]),
        #                         "c":Variation([0.1,0.01,0.001]),
        #                         "l":Variation(["MC", "NB"])
        #                     },
        #                     "s" : Variation([4,8,16,32,64]),
        #                     "o" : "(Percentage (M * (m / 100)))",
        #                     "x" : Variation(["(ADWINChangeDetector -a 1.0E-2)","(ADWINChangeDetector -a 1.0E-3)","(ADWINChangeDetector -a 1.0E-4)", "(ADWINChangeDetector -a 1.0E-5)"])
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
        #                 "moa_model":"moa.classifiers.meta.OnlineSmoothBoost",
        #                 "moa_params": {
        #                     "s" : Variation([4,8,16,32,64]),
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
        #                 "moa_model":"moa.classifiers.meta.AdaptiveRandomForest",
        #                 "moa_params": {
        #                     "s" : Variation([4,8,16,32,64]),
        #                     "j" : 1, # make sure to only use one thread for fairness
        #                     "x" : Variation(["(ADWINChangeDetector -a 1.0E-2)","(ADWINChangeDetector -a 1.0E-3)","(ADWINChangeDetector -a 1.0E-4)", "(ADWINChangeDetector -a 1.0E-5)"])
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
        #                 "moa_model":"moa.classifiers.meta.OzaBag",
        #                 "moa_params": {
        #                     "s" : Variation([4,8,16,32,64]),
        #                     "l": {
        #                         MoaModel.MOA_EMPTY_PLACEHOLDER : Variation(["moa.classifiers.trees.HoeffdingTree", "moa.classifiers.trees.EFDT"]), # 
        #                         "g":Variation([50,100,250]),
        #                         "c":Variation([0.1,0.01,0.001]),
        #                         "l":Variation(["MC", "NB"])
        #                     }
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
        #                     "g":Variation([50,100,250]),
        #                     "c":Variation([0.1,0.01,0.001]),
        #                     "l":Variation(["MC", "NB"])
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
        #                 "moa_model":"moa.classifiers.trees.HoeffdingTree",
        #                 "moa_params": {
        #                     "g":Variation([50,100,250]),
        #                     "c":Variation([0.1,0.01,0.001]),
        #                     "l":Variation(["MC", "NB"])
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
        #                 "moa_model":"moa.classifiers.bayes.NaiveBayes",
        #                 "nominal_attributes":nominal_attributes,
        #                 "moa_jar":os.path.join("moa-release-2020.12.0", "lib", "moa.jar"),
        #                 **online_learner_cfg
        #             },
        #             **experiment_cfg
        #         }, 
        #         n_configs=args.n_configs
        #     )
        # )

        random.shuffle(models)
        run_experiments(basecfg, models)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
    parser.add_argument("-d", "--dataset", help="Dataset used for experiments. Can be multiple entries",type=str, default=["gas-sensor"], nargs='+')
    parser.add_argument("-c", "--n_configs", help="Number of configs per base learner",type=int, default=1)
    parser.add_argument("-t", "--timeout", help="Maximum number of seconds per algorithm / dataset combination. If the runtime exceeds the provided value, stop execution of that single experiment",type=int, default=3600)
    parser.add_argument("-o", "--out_path", help="Path where results should be written",type=str, default=".")

    args = parser.parse_args()

    main(args)