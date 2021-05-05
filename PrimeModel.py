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
                additional_tree_options = {
                    "splitter" : "best", "criterion" : "gini"
                },
                eval_loss = "cross-entropy",
                shuffle = False,
                backend = "python"
        ):

        super().__init__(eval_loss, seed, verbose, shuffle, out_path)

        if "tree_init_mode" in additional_tree_options:
            assert additional_tree_options["tree_init_mode"] in ["train", "fully-random", "random"], "Currently only {{train, fully-random, random}} as tree_init_mode is supported"
            self.tree_init_mode = additional_tree_options["tree_init_mode"]
        else:
            self.tree_init_mode = "train^"

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
        self.max_depth = max_depth
        self.loss = loss
        self.step_size = step_size
        self.ensemble_regularizer = ensemble_regularizer
        self.l_ensemble_reg = l_ensemble_reg
        self.tree_regularizer = tree_regularizer
        self.l_tree_reg = l_tree_reg
        self.normalize_weights = normalize_weights
        self.init_weight = init_weight
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

    def num_parameters(self):
        if self.backend == "c++":
            return (2**(self.max_depth + 1) - 1)*self.num_trees()
        else:
            return self.model.num_parameters()

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

        # This is a little in-efficient since self.model.next already gives the output
        # and some statistics (depending on the backend). However, these statistics / output are for the entire batch and not the given example (data, target)
        # The c++ bindings only supports batched data and thus we add the implicit batch dimension via data[np.newaxis,:]
        output = np.array(self.model.predict_proba(data[np.newaxis,:]))[0]
        self.model.next(batch_data, batch_target)
        accuracy = (output.argmax() == target) * 100.0

        return {"accuracy": accuracy, "num_trees": self.num_trees(), "num_parameters" : self.num_parameters()}, output

    def fit(self, X, y, sample_weight = None):
        if self.backend == "c++":
            if self.init_weight in ["average", "max"]:
                weight_init_mode = self.init_weight
                init_weight = 1.0
            else:
                weight_init_mode = "constant"
                init_weight = self.init_weight
            
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

            self.model = CPrimeBindings(
                len(unique_labels(y)), 
                self.max_depth,
                self.seed,
                self.normalize_weights,
                self.loss,
                self.step_size,
                weight_init_mode,
                float(init_weight),
                is_nominal,
                ensemble_regularizer,
                float(self.l_ensemble_reg),
                tree_regularizer,
                float(self.l_tree_reg),
                self.tree_init_mode, 
                tree_update_mode
            )
        else:
            self.model = Prime(
                self.max_depth,
                self.loss,
                self.step_size,
                self.ensemble_regularizer,
                self.l_ensemble_reg,
                self.tree_regularizer,
                self.l_tree_reg,
                self.normalize_weights,
                self.init_weight,
                self.update_leaves,
                self.batch_size,
                self.verbose,
                self.out_path,
                self.seed,
                1,
                self.additional_tree_options
            )

            self.model.classes_ = sorted(unique_labels(y)) # TODO rework this for CPrime
            self.model.n_classes_ = len(self.model.classes_)
            self.model.n_outputs_ = self.model.n_classes_
            
            self.model.X_ = X
            self.model.y_ = y
        super().fit(X,y, sample_weight)