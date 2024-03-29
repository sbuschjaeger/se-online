import numpy as np
import random
import copy
from numpy.core.fromnumeric import argsort
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from scipy.special import softmax

from OnlineLearner import OnlineLearner

class RiverModel(OnlineLearner):
    def __init__(self, river_model, *args, **kwargs):

        assert river_model is not None, "river_model was None. This does not work!"
        
        # if "sliding_window" in args and args["sliding_window"] == False:
        #     print("WARNING: sliding_window should be True for RiverModel, but it was set to False. Fixing it for you.")
        #     args["sliding_window"] = True
        
        # if "sliding_window" in kwargs and kwargs["sliding_window"] == False:
        #     print("WARNING: sliding_window should be True for RiverModel, but it was set to False. Fixing it for you.")
        #     kwargs["sliding_window"] = True

        # if "batch_size" in args and args["batch_size"] >= 1:
        #     if args["batch_size"] > 1:
        #         print("WARNING: batch_size should be 1 for RiverModel for optimal performance, but was {}. Fixing it for you.".format(args["batch_size"]))
        #     args.pop("batch_size")

        # if "batch_size" in kwargs and kwargs["batch_size"] >= 1:
        #     if kwargs["batch_size"] > 1:
        #         print("WARNING: batch_size should be 1 for RiverModel for optimal performance, but was {}. Fixing it for you.".format(kwargs["batch_size"]))
        #     kwargs.pop("batch_size")

        super().__init__(*args, **kwargs)

        self.model = copy.deepcopy(river_model)
    
    def predict_proba_one(self, x):
        data = {}
        for j, xj in enumerate(x):
            data["att_" + str(j)] = xj

        output = self.model.predict_proba_one(data)
        pred = np.zeros(self.n_classes_)
        if len(output) == 0:
            return 1.0 / self.n_classes_ * np.ones(self.n_classes_)
        else:
            for key, val in output.items():
                pred[key] = val
        
        return pred

    def predict_proba(self, X):
        if len(X.shape) < 2:
            return self.predict_proba_one(X)
        else:
            proba = []
            for x in X:
                proba.append(self.predict_proba_one(x))
            return np.array(proba)

    def num_trees(self):
        if hasattr(self.model, "n_models"):
            return self.model.n_models
        else:
            return 1

    def num_parameters(self):
        n_nodes = 0
        # if hasattr(self.model, "models"):
        #     for m in self.model.models:
        #         if hasattr(m, "model"):
        #             n_nodes += 2 * m.model._n_decision_nodes + self.n_classes_ * m.model._n_active_leaves + (self.n_classes_ + 1) * m.model._n_inactive_leaves
        #         else:
        #             n_nodes += 2 * m._n_decision_nodes + self.n_classes_ * m._n_active_leaves + (self.n_classes_ + 1) * m._n_inactive_leaves
        # else:
        #     n_nodes = 2 * self.model._n_decision_nodes + self.n_classes_ * self.model._n_active_leaves + (self.n_classes_ + 1) * model._n_inactive_leaves
        if hasattr(self.model, "models"):
            for m in self.model.models:
                if hasattr(m, "model"):
                    n_nodes += m.model._n_decision_nodes + m.model._n_active_leaves + m.model._n_inactive_leaves
                else:
                    n_nodes += m._n_decision_nodes + m._n_active_leaves + m._n_inactive_leaves
        else:
            n_nodes = self.model._n_decision_nodes + self.model._n_active_leaves + self.model._n_inactive_leaves
        return n_nodes

    def next(self, data, target):
        # output = self.predict_proba_one(data)
        x_dict = {}
        for i, xi in enumerate(data):
            x_dict["att_" + str(i)] = xi
        self.model.learn_one(x_dict, target)

        # output = np.array(output)
        # accuracy = (output.argmax() == target) * 100.0

        # return {"accuracy": accuracy, "num_trees": self.num_trees(), "num_parameters" : self.num_parameters()}, output
