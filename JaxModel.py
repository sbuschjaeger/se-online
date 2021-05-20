import numpy as np
import random
from tqdm import tqdm

from absl import logging
# Disable "WARNING: Logging before flag parsing goes to stderr." message
# logging._warn_preinit_stderr = 0
# logging._warn_preinit_stdout = 0
logging.set_verbosity(logging.ERROR)

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
    to_expand = [[ [0, True] ]]
    pathes = []
    while to_expand:
        path = to_expand.pop()
        last_index = path[-1][0]
        if last_index > max_index:
            pathes.append(path)
        else:
            to_expand.append(path + [ [2*last_index + 1, True] ])
            to_expand.append(path + [ [2*last_index + 2, False] ])
    return pathes

# @jit 
# def tree_predict_proba(X, W, B, leaf_preds, beta = 1.0):
#     # print(beta)
#     # temp. scaling für sigmoid
#     #W /= jax.numpy.linalg.norm(W, ord = 2, axis=1)
#     all_preds = jax.nn.sigmoid( beta * ((W * X[:,np.newaxis,:]).sum(axis=2) + B.T) ) 
#     # all_preds = jax.nn.sigmoid( beta * ((W[np.newaxis,:,:] * X[:,np.newaxis,:]).sum(axis=2) + B.T) ) 
    
#     indices = path_indicies_iter(all_preds.shape[1] - 1)
#     #indices = path_indicies(all_preds.shape[1] - 1, [0])
#     def _pred(path):
#         leaf_node = path[-1][0]
#         #inner_nodes = path[:-1]
#         path_pred = 1.0
#         for pi, is_left in path:
#             if is_left:
#                 path_pred *= all_preds[:,pi]
#             else:
#                 path_pred *= (1.0 - all_preds[:,pi])

#         #path_preds = jax.numpy.prod(all_preds[:,inner_nodes], axis=1)
        
#         # p = 0.25
#         # dropout = np.random.choice(a=[True, False], p=[p, 1-p])
#         # if dropout:
#         #     path_preds = 0
#         # else:
#         #     path_preds /= (1-p)
#         # print(path_pred)
#         return (path_pred  * leaf_preds[leaf_node,:][:,np.newaxis]).T


#     # TODO add soft = true / false
#     indices = jax.numpy.array(indices)
#     # print(indices)
#     # print(indices.shape)
#     #preds = vmap(_pred)(indices)
#     preds = []
#     for path in indices:
#         preds.append(_pred(path))
    
#     # preds = vmap(_pred)(jax.numpy.array(indices))
#     # preds = []
#     # for path in indices:
#     #     leaf_node = path[-1]
#     #     inner_nodes = path[:-1]
#     #     path_preds = jax.numpy.prod(all_preds[:,inner_nodes], axis=1)
        
#     #     pred = (path_preds  * leaf_preds[leaf_node,:][:,np.newaxis]).T
#     #     preds.append(pred)
#     preds = jax.numpy.stack(preds)
#     #print(preds.sum(axis=0))
#     return preds.sum(axis=0)

@jit 
def tree_predict_proba(X, W, B, leaf_preds, beta = 1.0):
    # W = W / (jax.numpy.linalg.norm(W, ord = 2, axis=1, keepdims=True) + 1e-7)
    all_preds = jax.nn.sigmoid( beta * ((W * X[:,np.newaxis,:]).sum(axis=2) + B.T) ) 
    
    path_probs = []
    to_expand = [ (0, 1.0)  ]
    max_index = W.shape[0] - 1

    while to_expand:
        last_index, pprob = to_expand.pop()
        if last_index > max_index:
            path_probs.append(pprob[:,np.newaxis])
        else:
            to_expand.append( (2*last_index + 1, pprob * all_preds[:,last_index]) )
            to_expand.append( (2*last_index + 2, pprob * (1.0 - all_preds[:,last_index])) )

    path_probs = jax.numpy.stack(path_probs)
    preds = path_probs * leaf_preds[:,np.newaxis,:]
    #preds = path_probs * leaf_preds
    #print(preds.sum(axis=0))
    return preds.sum(axis=0)

# @jit
def predict_proba(X, W, B, leaf_preds, beta = 1.0):
    # TODO THIS SHOULD BE A pmap shouldnt it?
    ensemble_preds = []
    for w, b, l in zip(W,B,leaf_preds):
        preds = tree_predict_proba(X,w,b,l,beta)
        ensemble_preds.append(preds)

    return jax.numpy.stack(ensemble_preds).mean(axis=0)

class JaxModel(OnlineLearner):
    def __init__(self,  
                max_depth,
                n_trees = 1,
                step_size = 1e-3,
                loss = "cross-entropy",
                temp_scaling = 2,
                temp_start = 1,
                temp_max = 6,
                l_reg = 0,
                batch_size = 128,
                *args, **kwargs
                ):
                        
        assert loss in ["mse","cross-entropy"], "Currently only {mse, cross entropy} loss is supported"
        assert n_trees >= 1, "Num trees should be at-least 1"

        super().__init__(*args, **kwargs)

        self.max_depth = max_depth
        self.n_trees = n_trees
        self.step_size = step_size
        self.loss = loss
        self.beta = temp_start
        self.beta_max = temp_max
        self.temp_scaling = temp_scaling
        self.l_reg = l_reg
        self.batch_size = batch_size
        
        self.cur_batch_x = [] 
        self.cur_batch_y = [] 

    def predict_proba(self, X):
        if len(X.shape) < 2:
            X = X[np.newaxis,:]
            return np.array(predict_proba(X, self.W, self.B, self.leaf_preds, self.beta))
        else:
            return np.array(predict_proba(X, self.W, self.B, self.leaf_preds, self.beta))

    def num_bytes(self):
        # 4 byte?
        return self.num_trees() * (self.W[0].shape[0] * self.W[0].shape[1] + self.B[0].shape[0] + self.leaf_preds[0].shape[0] * self.leaf_preds[0].shape[1]) * 4

    def num_trees(self):
        return self.n_trees

    def num_nodes(self):
        return self.n_nodes

    def next(self, data, target):
        if len(self.cur_batch_x) > self.batch_size:
            self.cur_batch_x.pop(0)
            self.cur_batch_y.pop(0)

        self.cur_batch_x.append(data)
        self.cur_batch_y.append(target)
        
        batch_data = np.array(self.cur_batch_x)
        batch_target = np.array(self.cur_batch_y)

        # If train = True we basically call predict_proba twice
        # TODO Change this
        #output = self.predict_proba(batch_data)

        # TODO Activate this?
        # if new_epoch:
        #     self.beta = min(self.temp_scaling * self.beta, self.beta_max)

        def _loss(W, B, leaf_preds, x, beta):
            pred = predict_proba(x, W, B, leaf_preds, beta) #self.all_pathes, self.soft
            if self.loss == "mse":
                target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in batch_target] )
                loss = (pred - target_one_hot) * (pred - target_one_hot)
            elif self.loss == "cross-entropy":
                target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in batch_target] )
                p = jax.nn.softmax(pred, axis=1)
                loss = -target_one_hot*jax.numpy.log(p)
            
            # std_reg = []
            # for i in range(self.num_trees()):
            #     std_reg.append( jax.numpy.std(self.leaf_preds[i], axis=0) )
            # std_reg = jax.numpy.stack(std_reg)
            
            # loss = loss.mean() - std_reg.mean()
            loss = loss.mean()

            if self.l_reg > 0:
                w_reg = []
                b_reg = []
                for i in range(self.num_trees()):
                    w_reg.append( jax.numpy.linalg.norm(self.W[i], ord = 1, axis=1) )
                    b_reg.append( jax.numpy.linalg.norm(self.B[i], ord = 1, axis=1) )

                w_reg = jax.numpy.stack(w_reg)
                b_reg = jax.numpy.stack(b_reg)
                #return loss.mean() + self.l_reg * w_reg.mean() + self.l_reg * b_reg.mean()
                return loss + self.l_reg * w_reg.mean() + self.l_reg * b_reg.mean()
            else:
                return loss#.mean()

        loss, gradient = value_and_grad(_loss, (0, 1, 2))(self.W, self.B, self.leaf_preds, batch_data, self.beta)
        W_grad, B_grad, leaf_preds_grad = gradient

        for i in range(self.num_trees()):
            self.W[i] = self.W[i] - self.step_size * W_grad[i]
            self.B[i] = self.B[i] - self.step_size * B_grad[i]
            self.leaf_preds[i] = self.leaf_preds[i] - self.step_size * leaf_preds_grad[i]
            
            # print("GRAD LEAFs:", leaf_preds_grad[i])
            # print("LEAF:", self.leaf_preds[i])
            # print("W:", self.W[i])
            # print("B:", self.B[i])

        # accuracy = (output.argmax(axis=1) == batch_target) * 100.0
        # n_trees = [self.num_trees() for _ in range(batch_data.shape[0])]
        # n_param = [self.num_parameters() for _ in range(batch_data.shape[0])]
        # return {"accuracy": accuracy, "num_trees": n_trees, "num_parameters" : n_param}, output

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
