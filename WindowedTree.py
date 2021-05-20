import numpy as np
import random
import pickle
import sys
from numpy.core.fromnumeric import argsort
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from scipy.special import softmax

from OnlineLearner import OnlineLearner

from prime.Prime import Prime
from prime.CPrimeBindings import CPrimeBindings

class WindowedTree(OnlineLearner):
    """ 

    Attributes
    ----------
    max_depth : int
        Maximum depth of DTs trained on each batch
    step_size : float
        The step_size used for stochastic gradient descent for opt 
    loss : str
        The loss function for training. Should be one of `{"mse", "cross-entropy", "hinge2"}`
    normalize_weights : boolean
        True if nonzero weights should be projected onto the probability simplex, that is they should sum to 1. 
    ensemble_regularizer : str
        The ensemble_regularizer. Should be one of `{None, "L0", "L1", "hard-L1"}`
    l_ensemble_reg : float
        The ensemble_regularizer regularization strength. 
    tree_regularizer : str
        The tree_regularizer. Should be one of `{None,"node"}`
    l_tree_reg : float
        The tree_regularizer regularization strength. 
    init_weight : str, number
        The weight initialization for each new tree. If this is `"max`" then the largest weight across the entire ensemble is used. If this is `"average"` then the average weight  across the entire ensemble is used. If this is a number, then the supplied value is used. 
    batch_size: int
        The batch sized used for SGD
    update_leaves : boolean
        If true, then leave nodes of each tree are also updated via SGD.
    epochs : int
        The number of epochs SGD is run.
    verbose : boolean
        If true, shows a progress bar via tqdm and some statistics
    out_path: str
        If set, stores a file called epoch_$i.npy with the statistics for epoch $i under the given path.
    seed : None or number
        Random seed for tree construction. If None, then the seed 1234 is used.
    estimators_ : list of objects
        The list of estimators which are used to built the ensemble. Each estimator must offer a predict_proba method.
    estimator_weights_ : np.array of floats
        The list of weights corresponding to their respective estimator in self.estimators_. 
    """

    def __init__(self,
                max_depth,
                batch_size,
                splitter = "best",
                criterion = "gini",
                verbose = False,
                out_path = None,
                seed = None,
                eval_loss = "cross-entropy",
                shuffle = False
        ):

        super().__init__(eval_loss, seed, verbose, shuffle, out_path)

        self.max_depth = max_depth
        self.batch_size = batch_size
        self.criterion = criterion
        self.splitter = splitter

        self.model = None
        self.dt_seed = seed

        self.cur_batch_x = [] 
        self.cur_batch_y = [] 

    def num_trees(self):
        if self.model is None:
            return 0
        else:
            return 1

    def num_bytes(self):
        p = pickle.dumps(self.model)
        return sys.getsizeof(p) + sys.getsizeof(self.cur_batch_x) + sys.getsizeof(self.cur_batch_y)

    def num_nodes(self):
        if self.model is None:
            return 0
        else:
            return self.model.tree_.node_count

    def predict_proba(self, X):
        if self.model is None:
            return 1.0 / self.n_classes_ * np.ones((X.shape[0], self.n_classes_))
        else:
            proba = np.zeros(shape=(X.shape[0], self.n_classes_), dtype=np.float32)
            if len(X.shape) < 2:
                # add the implicit batch dimension via X[np.newaxis,:]
                X = X[np.newaxis,:]
            
            proba[:, self.model.classes_.astype(int)] += self.model.predict_proba(X)

            return proba

    def next(self, data, target):
        # The python and the c++ backend are both batched algorithms
        # and we want to also consume data in a sliding window approach.
        if len(self.cur_batch_x) > self.batch_size:
            self.cur_batch_x.pop(0)
            self.cur_batch_y.pop(0)

        self.cur_batch_x.append(data)
        self.cur_batch_y.append(target)
        
        batch_data = np.array(self.cur_batch_x)
        batch_target = np.array(self.cur_batch_y)

        # output = np.array(self.predict_proba(data[np.newaxis,:]))[0]
        
        self.model = DecisionTreeClassifier(max_depth = self.max_depth, random_state=self.dt_seed, splitter=self.splitter, criterion=self.criterion)
        self.dt_seed += 1
        self.model.fit(batch_data, batch_target)

        # accuracy = (output.argmax() == target) * 100.0

        # return {"accuracy": accuracy, "num_trees": self.num_trees(), "num_parameters" : self.num_parameters()}, output
