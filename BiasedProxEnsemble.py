import numpy as np
import random
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from scipy.special import softmax

from OnlineLearner import OnlineLearner
# from plotille import histogram
#from PyBPE import BiasedProxEnsemble
import PyBPE

class BiasedProxEnsemble(OnlineLearner):
    def __init__(self,  
                max_depth,
                max_trees = 0,
                step_size = 1e-1,
                l_reg = 1e-2,
                loss = "cross-entropy",
                init_mode = "random",
                next_mode = "incremental",
                init_weight = 0,
                *args, **kwargs
                ):
                        
        assert loss in ["mse","cross-entropy"], "Currently only {mse, cross entropy} loss is supported"
        assert init_mode in ["random", "train", "fully-random"], "Currently only {random, train, fully-random} init_mode supported"
        assert next_mode in ["incremental", "none", "gradient"], "Currently only {incremental, none, gradient} next_mode supported"
        assert max_depth >= 1, "max_depth should be at-least 1!"
        assert max_trees >= 0, "max_trees should be at-least 0!"
        
        super().__init__(*args, **kwargs)

        self.max_depth = max_depth
        self.max_trees = max_trees
        self.step_size = step_size
        self.l_reg = l_reg
        self.loss = loss
        self.init_mode = init_mode
        self.next_mode = next_mode
        self.init_weight = init_weight
        self.model = None
    
    def predict_proba(self, X):
        assert self.model is not None, "Call fit before calling predict_proba!"
        return np.array(self.model.predict_proba(X))

    def next(self, data, target, train = False, new_epoch = False):
        if train:
            lsum = self.model.next(data, target)
            output = self.predict_proba(data)
            return {"loss": lsum / (data.shape[0] * self.n_classes_), "num_trees": self.model.num_trees(), "num_parameters":self.num_parameters()}, output
        else:
            output = self.predict_proba(data)
            if self.loss == "mse":
                target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
                p = softmax(output, axis=1)
                loss = (p - target_one_hot) * (p - target_one_hot)
            elif self.loss == "cross-entropy":
                target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
                p = softmax(output, axis=1)
                loss = -target_one_hot*np.log(p + 1e-7)
            else: 
                raise "Currently only the three losses {{cross-entropy, exp}} are supported, but you provided: {}".format(self.loss)
            return {"loss": np.mean(loss), "num_trees": self.num_trees(), "num_parameters":self.num_parameters()}, output

    def num_trees(self):
        return self.model.num_trees()

    def num_parameters(self):
        n_inner = 2**(self.max_depth) - 1
        n_leafs = 2**(self.max_depth) 
        return self.model.num_trees() * self.n_classes_ * n_leafs #(2*n_inner + self.n_classes_ * n_leafs)

    def fit(self, X, y, sample_weight = None):
        classes_ = unique_labels(y)
        n_classes_ = len(classes_)
        self.model = PyBPE.BiasedProxEnsemble(self.max_depth, self.max_trees, n_classes_, self.seed, self.step_size, self.l_reg, self.init_weight, self.init_mode, self.next_mode, self.loss)
        super().fit(X, y, sample_weight)
