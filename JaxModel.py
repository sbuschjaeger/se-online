import numpy as np
import random
from tqdm import tqdm

import jax
from jax import grad
from jax import value_and_grad
from jax import jit
from jax import vmap
from jax import pmap

from functools import partial

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from scipy.special import softmax
from OnlineLearner import OnlineLearner

@jax.partial(jit, static_argnums=0) 
def path_indicies(max_depth, path):
    i = path[-1]

    if (i > max_depth):
        return [path]
    else:
        lpath = path + [2*i+1]
        lpath = path_indicies(max_depth, lpath)
        
        rpath = path + [2*i+2]
        rpath = path_indicies(max_depth, rpath)

        lpath.extend(rpath)
        return lpath

@jax.partial(jit, static_argnums=0)
def path_indicies_iter(max_index):
    to_expand = [[0]]
    pathes = []
    while to_expand:
        path = to_expand.pop()
        if path[-1] > max_index:
            pathes.append(path)
        else:
            to_expand.append(path + [2*path[-1] + 1])
            to_expand.append(path + [2*path[-1] + 2])
    return pathes

@jit 
def tree_predict_proba(X, W, B, leaf_preds):
    all_preds = jax.nn.sigmoid( (W * X[:,np.newaxis,:]).sum(axis=2) + B.T ) 
    
    indices = path_indicies_iter(all_preds.shape[1] - 1)
    #indices = path_indicies(all_preds.shape[1] - 1, [0])
    def _pred(path):
        leaf_node = path[-1]
        inner_nodes = path[:-1]
        path_preds = jax.numpy.prod(all_preds[:,inner_nodes], axis=1)
        
        return (path_preds  * leaf_preds[leaf_node,:][:,np.newaxis]).T

    indices = jax.numpy.array(indices)
    preds = vmap(_pred)(jax.numpy.array(indices))
    # preds = []
    # for path in indices:
    #     leaf_node = path[-1]
    #     inner_nodes = path[:-1]
    #     path_preds = jax.numpy.prod(all_preds[:,inner_nodes], axis=1)
        
    #     pred = (path_preds  * leaf_preds[leaf_node,:][:,np.newaxis]).T
    #     preds.append(pred)
    preds = jax.numpy.stack(preds)
    return preds.sum(axis=0)

@jit
def predict_proba(X, W, B, leaf_preds):
    # TODO THIS SHOULD BE A pmap shouldnt it?
    ensemble_preds = []
    for w, b, l in zip(W,B,leaf_preds):
        preds = tree_predict_proba(X,w,b,l)
        ensemble_preds.append(preds)

    return jax.numpy.stack(ensemble_preds).mean(axis=0)

class JaxModel(OnlineLearner):
    def __init__(self,  
                max_depth,
                n_trees = 1,
                step_size = 1e-3,
                loss = "cross-entropy",
                *args, **kwargs
                ):
                        
        assert loss in ["mse","cross-entropy"], "Currently only {mse, cross entropy} loss is supported"
        assert n_trees >= 1, "Num trees should be at-least 1"

        super().__init__(*args, **kwargs)

        self.max_depth = max_depth
        self.n_trees = n_trees
        self.step_size = step_size
        self.loss = loss

    def predict_proba(self, X):
        return predict_proba(X, self.W, self.B, self.leaf_preds)

    def num_trees(self):
        return self.n_trees

    def num_nodes(self):
        return self.num_trees() * (2**(self.max_depth + 1) - 1)

    def next(self, data, target, train = False):
        if train:
            def _loss(W, B, leaf_preds, x):
                pred = predict_proba(x, W, B, leaf_preds) #self.all_pathes, self.soft
                if self.loss == "mse":
                    target_one_hot = np.array( [ [1 if y == i else 0 for i in range(self.n_classes_)] for y in target] )
                    loss = (pred - target_one_hot) * (pred - target_one_hot)
                elif self.loss == "cross-entropy":
                    target_one_hot = np.array( [ [1 if y == i else 0 for i in range(self.n_classes_)] for y in target] )
                    p = jax.nn.softmax(pred, axis=1)
                    loss = -target_one_hot*jax.numpy.log(p)
                return loss.mean()

            loss, gradient = value_and_grad(_loss, (0, 1, 2))(self.W, self.B, self.leaf_preds, data)
            W_grad, B_grad, leaf_preds_grad = gradient

            for i in range(self.num_trees()):
                self.W[i] = self.W[i] - self.step_size * W_grad[i]
                self.B[i] = self.B[i] - self.step_size * B_grad[i]
                self.leaf_preds[i] = self.leaf_preds[i] - self.step_size * leaf_preds_grad[i]
            
            output = self.predict_proba(data)

            return {"loss": np.asarray(loss), "num_trees": self.num_trees(),"num_nodes":self.num_nodes()}, output
        else:
            output = self.predict_proba(data)
            if self.loss == "mse":
                target_one_hot = np.array( [ [1 if y == i else 0 for i in range(self.n_classes_)] for y in target] )
                loss = (output - target_one_hot) * (output - target_one_hot)
            elif self.loss == "cross-entropy":
                target_one_hot = np.array( [ [1 if y == i else 0 for i in range(self.n_classes_)] for y in target] )
                p = jax.nn.softmax(output, axis=1)
                loss = -target_one_hot*jax.numpy.log(p)
            return {"loss": np.asarray(loss.mean()), "num_trees": self.num_trees()}, output

    def fit(self, X, y, sample_weight = None):
        classes_ = unique_labels(y)
        n_classes_ = len(classes_)
        self.inner_weights = []
        self.leaf_weights = []
        self.n_nodes = 2**(self.max_depth + 1) - 1
        self.n_leafs = 2**self.max_depth #- 1
        self.n_inner = self.n_nodes - self.n_leafs
        key = jax.random.PRNGKey(self.seed)

        self.W = [jax.random.normal(key, (self.n_inner, X.shape[1])) for _ in range(self.num_trees())]
        self.B = [jax.random.normal(key, (self.n_inner, 1)) for _ in range(self.num_trees())]
        self.leaf_preds = [jax.random.uniform(key, (self.n_leafs, n_classes_)) for _ in range(self.num_trees())]
        super().fit(X,y,sample_weight)
