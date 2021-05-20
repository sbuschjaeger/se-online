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
from prime.CPrimeBindings import CPrimeBindings

class PrimeModel(OnlineLearner):
    """ 

    Attributes
    ----------
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
                loss = "cross-entropy",
                step_size = 1e-1,
                ensemble_regularizer = None,
                l_ensemble_reg = 0,  
                tree_regularizer = None,
                l_tree_reg = 0,
                normalize_weights = False,
                update_leaves = False,
                batch_size = 256,
                verbose = False,
                out_path = None,
                seed = None,
                additional_tree_options = {
                    "splitter" : "best", 
                    "criterion" : "gini", 
                    "max_depth": None
                },
                eval_loss = "cross-entropy",
                shuffle = False,
                backend = "python"
        ):

        super().__init__(eval_loss, seed, verbose, shuffle, out_path)

        if backend == "c++":
            assert "max_depth" in additional_tree_options and additional_tree_options["max_depth"] > 0, "The C++ backend required a maximum tree depth to be set, but none was given"

        if "tree_init_mode" in additional_tree_options:
            assert additional_tree_options["tree_init_mode"] in ["train", "fully-random", "random"], "Currently only {{train, fully-random, random}} as tree_init_mode is supported"
            self.tree_init_mode = additional_tree_options["tree_init_mode"]
        else:
            self.tree_init_mode = "train"

        if "is_nominal" in additional_tree_options:
            self.is_nominal = additional_tree_options["is_nominal"]
        else:
            self.is_nominal = None

        if seed is None:
            self.seed = 1234
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.backend = backend
        self.loss = loss
        self.step_size = step_size
        self.ensemble_regularizer = ensemble_regularizer
        self.l_ensemble_reg = l_ensemble_reg
        self.tree_regularizer = tree_regularizer
        self.l_tree_reg = l_tree_reg
        self.normalize_weights = normalize_weights
        self.update_leaves = update_leaves
        self.batch_size = batch_size
        self.verbose = verbose
        self.out_path = out_path
        self.seed = seed
        self.additional_tree_options = additional_tree_options
        self.model = None 

        self.cur_batch_x = [] 
        self.cur_batch_y = [] 

    def num_trees(self):
        return self.model.num_trees()

    def num_nodes(self):
        return self.model.num_nodes()

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

        self.model.next(batch_data, batch_target)
        # output = np.array(self.model.predict_proba(data[np.newaxis,:]))[0]
        # accuracy = (output.argmax() == target) * 100.0

        # return {"accuracy": accuracy, "num_trees": self.num_trees(), "num_nodes" : self.num_nodes()}, output

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def fit(self, X, y, sample_weight = None):
        self.model = Prime(
            self.loss,
            self.step_size,
            self.ensemble_regularizer,
            self.l_ensemble_reg,
            self.tree_regularizer,
            self.l_tree_reg,
            self.normalize_weights,
            self.update_leaves,
            self.batch_size,
            self.verbose,
            self.out_path,
            self.seed,
            1,
            self.backend,
            self.additional_tree_options
        )

        # Since we do not call "fit" of the Prime we have to prepare everything manually here
        if self.backend == "c++":
            if self.step_size == "adaptive":
                step_size_mode = "adaptive"
                step_size = 0
            else:
                step_size_mode = "constant"
                step_size = float(self.step_size)

            if self.update_leaves:
                tree_update_mode = "gradient"
            else:
                tree_update_mode = "none"
            
            if self.is_nominal is None:
                is_nominal = [False for _ in range(X.shape[1])]
            else:
                is_nominal = self.is_nominal

            ensemble_regularizer = "none" if self.ensemble_regularizer is None else str(self.ensemble_regularizer)
            tree_regularizer = "none" if self.tree_regularizer is None else str(self.tree_regularizer)

            self.model.model = CPrimeBindings(
                len(unique_labels(y)), 
                self.additional_tree_options["max_depth"],
                self.seed,
                self.normalize_weights,
                self.loss,
                step_size,
                step_size_mode,
                is_nominal,
                ensemble_regularizer,
                float(self.l_ensemble_reg),
                tree_regularizer,
                float(self.l_tree_reg),
                self.tree_init_mode, 
                tree_update_mode
            )
        
        self.model.classes_ = unique_labels(y)
        self.model.n_classes_ = len(self.model.classes_)
        self.model.n_outputs_ = self.model.n_classes_
        
        super().fit(X,y, sample_weight)