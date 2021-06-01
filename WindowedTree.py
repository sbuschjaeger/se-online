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

from prime.CPrimeBindings import CTreeBindings

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
                batch_size,
                additional_tree_options,
                verbose = False,
                out_path = None,
                seed = None,
                eval_loss = "cross-entropy",
                shuffle = False,
                backend = "c++"
        ):

        super().__init__(eval_loss, seed, verbose, shuffle, out_path)

        self.batch_size = batch_size

        self.additional_tree_options = additional_tree_options
        self.model = None
        self.dt_seed = seed
        self.backend = backend

        self.cur_batch_x = [] 
        self.cur_batch_y = [] 

    def num_trees(self):
        if self.model is None:
            return 0
        else:
            return 1

    def num_bytes(self):
        size = super().num_bytes()
        size += sys.getsizeof(self.cur_batch_x) + sys.getsizeof(self.cur_batch_y) + sys.getsizeof(self.additional_tree_options) + sys.getsizeof(self.batch_size) + sys.getsizeof(self.dt_seed) + sys.getsizeof(self.backend)
        
        if self.model is not None:
            if self.backend == "python":
                # The simplest way to get the size of an sklearn object is to pickel it. This includes a lot of overhead now due to python, but since the "backend" for computing the model is python in this case I guess this is a fair comparison. 
                p = pickle.dumps(self.model)
                size += sys.getsizeof(p) 
            else:
                size += self.model.num_bytes()

        return size

    def num_nodes(self):
        if self.model is None:
            return 0
        else:
            if self.backend == "python":
                return self.model.tree_.node_count
            else:
                return self.model.num_nodes()

    def predict_proba(self, X):
        if self.model is None:
            if len(X.shape) < 0:
                return 1.0 / self.n_classes_ * np.ones((1, self.n_classes_))
            else:
                return 1.0 / self.n_classes_ * np.ones((X.shape[0], self.n_classes_))
        else:
            if len(X.shape) < 2:
                proba = np.zeros(shape=(1, self.n_classes_), dtype=np.float32)
                # add the implicit batch dimension
                X = X[np.newaxis,:]
            else:
                proba = np.zeros(shape=(X.shape[0], self.n_classes_), dtype=np.float32)
            
            if self.backend == "python":
                classes = self.model.classes_.astype(int)
            else:
                classes = np.array([i for i in range(self.n_classes_)])

            proba[:, classes] += self.model.predict_proba(X)

            return proba

    def next(self, data, target):
        # The python and the c++ backend are both batched algorithms
        # and we want to also consume data in a sliding window approach.
        if len(self.cur_batch_x) > self.batch_size:
            self.cur_batch_x.pop(0)
            self.cur_batch_y.pop(0)

        self.cur_batch_x.append(data)
        self.cur_batch_y.append(target)
        
        if self.model is None or self.predict_proba(data).argmax(axis=1)[0] != target:
            batch_data = np.array(self.cur_batch_x)
            batch_target = np.array(self.cur_batch_y)

            if self.backend == "python":
                self.model = DecisionTreeClassifier(random_state=self.dt_seed, **self.additional_tree_options)
                self.model.fit(batch_data, batch_target)
            else:
                max_depth = int(self.additional_tree_options.get("max_depth", 1))
                tree_init = self.additional_tree_options.get("tree_init_mode", "train")
                self.model = CTreeBindings(max_depth = max_depth, n_classes = self.n_classes_, seed = self.dt_seed, X = batch_data, Y = batch_target, tree_init_mode = tree_init, tree_update_mode = "none")

            self.dt_seed += 1
