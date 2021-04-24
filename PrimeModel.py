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

from prime.Prime import Prime

class PrimeModel(OnlineLearner):
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
                loss = "cross-entropy",
                step_size = 1e-1,
                ensemble_regularizer = None,
                l_ensemble_reg = 0,  
                tree_regularizer = None,
                l_tree_reg = 0,
                normalize_weights = False,
                init_weight = "average",
                update_leaves = False,
                batch_size = 256,
                verbose = False,
                out_path = None,
                seed = None,
                epochs = None,
                additional_tree_options = {
                    "splitter" : "best", "criterion" : "gini"
                },
                eval_every_epochs = None,
                eval_loss = "cross-entropy",
                sliding_window = False,
                shuffle = False
        ):

        super().__init__(eval_loss, batch_size, sliding_window, epochs, seed, verbose, shuffle, out_path, eval_every_epochs)

        self.model = Prime(
            max_depth,
            loss,
            step_size,
            ensemble_regularizer,
            l_ensemble_reg,
            tree_regularizer,
            l_tree_reg,
            normalize_weights,
            init_weight,
            update_leaves,
            batch_size,
            verbose,
            out_path,
            seed,
            epochs,
            additional_tree_options
        )

    def num_trees(self):
        return self.model.num_trees()

    def num_parameters(self):
        return self.model.num_parameters()

    def next(self, data, target, train = False, new_epoch = False):
        return self.model.next(data, target)

    def fit(self, X, y, sample_weight = None):
        self.model.classes_ = unique_labels(y)
        self.model.n_classes_ = len(self.model.classes_)
        self.model.n_outputs_ = self.model.n_classes_
        
        self.model.X_ = X
        self.model.y_ = y
        super().fit(X,y, sample_weight)