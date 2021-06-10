import numpy as np
import pickle
import sys
from tqdm import tqdm

import time

from absl import logging
# Disable "WARNING: Logging before flag parsing goes to stderr." message
# logging._warn_preinit_stderr = 0
# logging._warn_preinit_stdout = 0
logging.set_verbosity(logging.ERROR)

import torch
from torch import nn

# import jax
# from jax import grad
# from jax import value_and_grad
# from jax import jit
# from jax import vmap
# from jax import pmap

from functools import partial

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from scipy.special import softmax
from OnlineLearner import OnlineLearner


# # @jit 
# def get_all_path_preds(X, W, B, beta = 1.0):
#     return torch.sigmoid( beta * ((W * X).sum(axis=2) + B.T) ) 

# # @jit 
# def tree_predict_proba(X, W, B, leaf_preds, beta = 1.0):
#     # W = W / (jax.numpy.linalg.norm(W, ord = 2, axis=1, keepdims=True) + 1e-7)
#     #all_preds = jax.nn.sigmoid( beta * ((W * X[:,np.newaxis,:]).sum(axis=2) + B.T) ) 
#     #all_preds = jax.nn.sigmoid( beta * ((W * X).sum(axis=2) + B.T) ) 
#     all_preds = get_all_path_preds(X, W, B, beta)
    
#     to_expand = [ (0, 1.0)  ]
#     max_index = W.shape[0] - 1
#     path_probs = []
#     while to_expand:
#         last_index, pprob = to_expand.pop()
#         if last_index > max_index:
#             path_probs.append(pprob[:,np.newaxis])
#         else:
#             to_expand.append( (2*last_index + 1, pprob * all_preds[:,last_index]) )
#             to_expand.append( (2*last_index + 2, pprob * (1.0 - all_preds[:,last_index])) )

#     path_probs = torch.stack(path_probs)
#     #preds = path_probs * leaf_preds[:,np.newaxis,:]

#     preds = path_probs * leaf_preds
#     preds = preds.sum(axis=0).unsqueeze(0)#[torch.newaxis,:]
#     return preds

# @jit
# def tensor_predict_proba(X, W, B, leaf_preds, beta = 1.0):
#     X = X[:,np.newaxis,:]
#     X = torch.tensor(X, requires_grad=False)
#     tmp = tree_predict_proba(X,W,B,leaf_preds,beta) 
#     return tmp
    # all_preds = torch.stack( [
    #     tree_predict_proba(X,w,b,l,beta) for w, b, l in zip(W,B,leaf_preds)
    #     #tree_predict_proba(X,W[0],B[0],leaf_preds[0],beta) 
    # ])
    # return all_preds.mean(axis=0)

def tree_predict_proba(X,W,B,leaf_preds,beta):
        # W = W / (jax.numpy.linalg.norm(W, ord = 2, axis=1, keepdims=True) + 1e-7)
        #all_preds = jax.nn.sigmoid( beta * ((W * X[:,np.newaxis,:]).sum(axis=2) + B.T) ) 
        #all_preds = jax.nn.sigmoid( beta * ((W * X).sum(axis=2) + B.T) ) 
        all_preds = torch.sigmoid( beta * ((W * X).sum(axis=2) + B.T) )
        
        to_expand = [ (0, 1.0)  ]
        max_index = W.shape[0] - 1
        path_probs = []
        while to_expand:
            last_index, pprob = to_expand.pop()
            if last_index > max_index:
                path_probs.append(pprob[:,np.newaxis])
            else:
                to_expand.append( (2*last_index + 1, pprob * all_preds[:,last_index]) )
                to_expand.append( (2*last_index + 2, pprob * (1.0 - all_preds[:,last_index])) )

        path_probs = torch.stack(path_probs)
        #preds = path_probs * leaf_preds[:,np.newaxis,:]

        preds = path_probs * leaf_preds
        preds = preds.sum(axis=0).unsqueeze(0)#[torch.newaxis,:]
        return preds

class TorchModel(OnlineLearner, nn.Module):
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

        # Lets keep it fair for the rest
        torch.set_num_threads(1)

        OnlineLearner.__init__(self,*args, **kwargs)
        nn.Module.__init__(self)
        
        # super(OnlineLearner, self).__init__(*args, **kwargs)
        # super(nn.Module, self).__init__()
        
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
            return np.array(self.tensor_predict_proba(X).detach())
        else:
            return np.array(self.tensor_predict_proba(X).detach())

    def num_bytes(self):
        size = super().num_bytes()
        size += sys.getsizeof(self.cur_batch_x) + sys.getsizeof(self.cur_batch_y) + sys.getsizeof(self.max_depth) + sys.getsizeof(self.batch_size) + sys.getsizeof(self.beta_max) + sys.getsizeof(self.temp_scaling) + sys.getsizeof(self.l_reg) + sys.getsizeof(self.beta) + sys.getsizeof(self.n_trees) + sys.getsizeof(self.step_size) + sys.getsizeof(self.loss)
        
        return size + sum([sys.getsizeof(w.storage()) for w in self.W]) + sum([sys.getsizeof(lp.storage()) for lp in self.leaf_preds]) + sum([sys.getsizeof(b.storage()) for b in self.B])

    def tensor_predict_proba(self, X):
        X = X[:,np.newaxis,:]
        X = torch.tensor(X, requires_grad=False)
        all_preds = torch.stack( [
            tree_predict_proba(X,w,b,l,self.beta) for w, b, l in zip(self.W,self.B,self.leaf_preds)
        ])
        return all_preds.mean(axis=1)

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

        pred = self.tensor_predict_proba(batch_data) #self.all_pathes, self.soft
        if self.loss == "mse":
            target_one_hot = torch.tensor( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in batch_target] )
            loss = (pred - target_one_hot) * (pred - target_one_hot)
        elif self.loss == "cross-entropy":
            target_one_hot = torch.tensor( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in batch_target] )
            p = torch.softmax(pred, axis=1)
            loss = -target_one_hot*torch.log(p)
        
        loss = loss.mean()

        if self.l_reg > 0:
            w_reg = []
            b_reg = []
            for i in range(self.num_trees()):
                w_reg.append( torch.linalg.norm(self.W[i], ord = 1, axis=1) )
                b_reg.append( torch.linalg.norm(self.B[i], ord = 1, axis=1) )

            w_reg = torch.stack(w_reg)
            b_reg = torch.stack(b_reg)
            #return loss.mean() + self.l_reg * w_reg.mean() + self.l_reg * b_reg.mean()
            loss += self.l_reg * w_reg.mean() + self.l_reg * b_reg.mean()
        
        loss.backward()
        self.optimizer.step()

        # with torch.no_grad():
        for i in range(self.num_trees()):
            # self.W[i] -= self.step_size * self.W[i].grad
            # self.B[i] -= self.step_size * self.B[i].grad
            # self.leaf_preds[i] -= self.step_size * self.leaf_preds[i].grad
            self.W[i].grad.zero_()
            self.B[i].grad.zero_()
            self.leaf_preds[i].grad.zero_()

    def fit(self, X, y, sample_weight = None):
        classes_ = unique_labels(y)
        n_classes_ = len(classes_)
        self.inner_weights = []
        self.leaf_weights = []
        self.n_nodes = 2**(self.max_depth + 1) - 1
        self.n_leafs = 2**self.max_depth #- 1
        self.n_inner = self.n_nodes - self.n_leafs
        #key = jax.random.PRNGKey(self.seed)

        self.W = [torch.rand((self.n_inner, X.shape[1]), requires_grad=True) for _ in range(self.num_trees())]
        self.B = [torch.rand((self.n_inner, 1), requires_grad=True) for _ in range(self.num_trees())]
        self.leaf_preds = [torch.rand((self.n_leafs,1, n_classes_), requires_grad=True) for _ in range(self.num_trees())]
        self.optimizer = torch.optim.SGD(self.W + self.B + self.leaf_preds, lr=self.step_size)
        # self.W = torch.rand((self.n_inner, X.shape[1]), requires_grad=True) 
        # self.B = torch.rand((self.n_inner, 1), requires_grad=True) 
        # self.leaf_preds = torch.rand((self.n_leafs,1, n_classes_), requires_grad=True)
        super().fit(X,y,sample_weight)
