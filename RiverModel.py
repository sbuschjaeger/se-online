import numpy as np
import random
import copy
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from scipy.special import softmax

from OnlineLearner import OnlineLearner

class RiverModel(OnlineLearner):
    def __init__(self, river_model, loss = "cross-entropy", *args, **kwargs):

        assert loss in ["mse","cross-entropy"], "Currently only {{mse, cross-entropy}} loss is supported"
        assert river_model is not None, "river_model was None. This does not work!"

        super().__init__(*args, **kwargs)

        self.model = copy.deepcopy(river_model)
        self.loss = loss
    
    def predict_proba_one(self, x):
        data = {}
        for j, xj in enumerate(x):
            data["att_" + str(j)] = xj

        output = self.model.predict_proba_one(data)
        pred = np.zeros(self.n_classes_)
        for key, val in output.items():
            pred[key] = val
        
        return pred

    def predict_proba(self, X):
        proba = []
        if (len(X) == 1):
            proba.append(self.predict_proba_one(X))
        else:
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
        if hasattr(self.model, "models"):
            for m in self.model.models:
                if hasattr(m, "model"):
                    n_nodes += 2 * m.model._n_decision_nodes + self.n_classes_ * m.model._n_active_leaves + (self.n_classes_ + 1) * m.model._n_inactive_leaves
                else:
                    n_nodes += 2 * m._n_decision_nodes + self.n_classes_ * m._n_active_leaves + (self.n_classes_ + 1) * m._n_inactive_leaves
        else:
            n_nodes = 2 * self.model._n_decision_nodes + self.n_classes_ * self.model._n_active_leaves + (self.n_classes_ + 1) * m.model._n_inactive_leaves
        return n_nodes

    def next(self, data, target, train = False, new_epoch = False):
        losses = []
        output = []
        for x, y in zip(data, target):
            if train:
                x_dict = {}
                for i, xi in enumerate(x):
                    x_dict["att_" + str(i)] = xi
                self.model.learn_one(x_dict, y)

            pred = self.predict_proba_one(x)
            
            if self.loss == "mse":
                target_one_hot = np.array( [1.0 if y == i else 0.0 for i in range(self.n_classes_)] )
                loss = (pred - target_one_hot) * (pred - target_one_hot)
            elif self.loss == "cross-entropy":
                target_one_hot = np.array( [1.0 if y == i else 0.0 for i in range(self.n_classes_)] )
                p = softmax(pred)
                loss = -target_one_hot*np.log(p)
            
            losses.append(loss)
            output.append(pred)
        
        return {"loss": np.mean(loss), "num_trees": self.num_trees(), "num_parameters":self.num_parameters()}, np.array(output), data.shape[0]
