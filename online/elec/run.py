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
from scipy.io.arff import loadarff

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import river
from river import tree
from river.ensemble import AdaptiveRandomForestClassifier

from JaxModel import JaxModel

from experiment_runner.experiment_runner import run_experiments, Variation, generate_configs

# from BiasedProxEnsemble import BiasedProxEnsemble
# from SGDEnsemble import SGDEnsemble
from RiverModel import RiverModel
from PyBiasedProxEnsemble import PyBiasedProxEnsemble

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
        "backend": "single",
        "verbose":True
    }
elif args.multi:
    basecfg = {
        "out_path":"results/" + datetime.now().strftime('%d-%m-%Y-%H:%M:%S'),
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "multiprocessing",
        "num_cpus":8,
        "verbose":True
    }
else:
    exit(1)

print("Loading data")
data, meta = loadarff("elecNormNew.arff")

print("Mapping nominal attributes")
Xdict = {}
for cname, ctype in zip(meta.names(), meta.types()):
    if cname == "class":
        enc = LabelEncoder()
        Xdict["label"] = enc.fit_transform(data[cname])
    elif ctype == "numeric":
        Xdict[cname] = data[cname]
    else:
        enc = OneHotEncoder(sparse=False)
        tmp = enc.fit_transform(data[cname].astype('<f8').reshape(-1, 1))
        for i in range(tmp.shape[1]):
            Xdict[cname + "_" + str(i)] = tmp[:,i]

df = pd.DataFrame(Xdict)
Y = df["label"].values.astype(np.int32)
df = df.drop("label", axis=1)
is_nominal = (df.nunique() == 2).values
nominal_names = [name for nom,name in zip(is_nominal, df.columns.values) if nom ]

scaler = MinMaxScaler()
X = scaler.fit_transform(df.values.astype(np.float64))
# np.save("X.npy", X, allow_pickle=True)
# np.save("Y.npy", Y, allow_pickle=True)
# np.save("nominal_names.npy", nominal_names, allow_pickle=True)
print(X.shape)

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
    "verbose":args.single,
    "shuffle":False,
    "eval_every_epochs":1   
}

n_configs = 50
np.random.seed(experiment_cfg["seed"])

models = []
print("Generating random hyperparameter configurations")

models.extend(
    generate_configs(
        {
            "model":PyBiasedProxEnsemble,
            "model_params": {
                "loss":"cross-entropy",
                "ensemble_regularizer":"hard-L1",
                "tree_regularizer":None,
                "l_tree_reg":0,
                "normalize_weights":True,
                "init_weight":"average",
                "seed":experiment_cfg["seed"],
                "batch_size":Variation([8,16,32,64,128,256]),
                "max_depth":Variation([2,3,4,5,6,7,8,9,10]),
                "step_size":Variation([1e-3,1e-2,1e-1,5e-1,1,2]),
                "l_ensemble_reg":Variation([16,32,64,128,256]),
                "update_trees":Variation([True, False]),
                **online_learner_cfg
            },
            **experiment_cfg
        }, 
        n_configs=n_configs
    )
)

models.extend(
    generate_configs(
        {
            "model":JaxModel,
            "model_params": {
                "loss":"cross-entropy",
                "step_size":Variation([1e-3,1e-2,1e-1,5e-1,1,2]),
                "max_depth":Variation([2,3,4,5]),
                "n_trees":Variation([1,2,4,8]),
                "batch_size":Variation([8,16,32,64,128,256]),
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
    )
)

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
                    "nominal_attributes" : nominal_names
                },
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
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
                    "nominal_attributes" : nominal_names
                },
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
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
                        "nominal_attributes" : nominal_names
                    }
                },
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
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
                        "nominal_attributes" : nominal_names
                    }
                },
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
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
                        "nominal_attributes" : nominal_names
                    }
                },
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
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
                        "nominal_attributes" : nominal_names
                    }
                },
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
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
                        "nominal_attributes" : nominal_names
                    }
                },
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
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
                        "nominal_attributes" : nominal_names
                    }
                },
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
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
                    "nominal_attributes" : nominal_names
                },
                **online_learner_cfg
            },
            **experiment_cfg
        },
        n_configs=n_configs
    )
)

random.shuffle(models)

run_experiments(basecfg, models)
