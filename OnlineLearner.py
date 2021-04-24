import os
import numpy as np
import random
from tqdm import tqdm
import json
import time

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

from abc import ABC, abstractmethod

# Modified from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
def create_mini_batches(inputs, targets, batch_size, shuffle=False, sliding_window=False):
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        
        if sliding_window:
            start_idx += 1
        else:
            start_idx += batch_size

        yield inputs[excerpt], targets[excerpt]

class OnlineLearner(ABC):
    def __init__(self,  
                eval_loss = "cross-entropy",
                batch_size = 256,
                sliding_window = False,
                epochs = None,
                seed = None,
                verbose = True, 
                shuffle = True,
                out_path = None,
                eval_every_epochs = None):
        
        assert eval_loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert epochs >= 1, "epochs must be at-least 1"
        assert eval_every_epochs is None or eval_every_epochs > 0, "eval_every epochs should either be None (nothing is stored) or > 0"

        self.eval_loss = eval_loss
        self.batch_size = batch_size
        self.sliding_window = sliding_window
        self.epochs = epochs
        self.verbose = verbose
        self.shuffle = shuffle
        self.eval_every_epochs = eval_every_epochs
        self.out_path = out_path

        if seed is None:
            self.seed = 1234
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)

    @abstractmethod
    def next(self, data, target, train=False, new_epoch = False):
        pass
    
    @abstractmethod
    def num_trees(self):
        pass
    
    @abstractmethod
    def num_parameters(self):
        pass

    def loss_(self, output, target):
        if self.eval_loss == "mse":
            target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
            loss = (output - target_one_hot) * (output - target_one_hot)
        elif self.eval_loss == "cross-entropy":
            target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
            p = softmax(output, axis=1)
            loss = -target_one_hot*np.log(p + 1e-7)
        elif self.eval_loss == "hinge2":
            target_one_hot = np.array( [ [1.0 if y == i else -1.0 for i in range(self.n_classes_)] for y in target] )
            zeros = np.zeros_like(target_one_hot)
            loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
        else:
            raise "Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.eval_loss)
        return loss

    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_
        
        self.X_ = X
        self.y_ = y

        for epoch in range(self.epochs):
            mini_batches = create_mini_batches(X, y, self.batch_size, self.shuffle, self.sliding_window) 

            metrics = {}
            metrics = {}

            new_epoch = epoch > 0 
            first_batch = True
            example_cnt = 0

            with tqdm(total=X.shape[0], ncols=150, disable = not self.verbose) as pbar:
                for batch in mini_batches: 
                    data, target = batch 
                    
                    # Update Model                    
                    start_time = time.time()
                    batch_metrics, output = self.next(data, target, train = True, new_epoch = new_epoch)
                    batch_time = time.time() - start_time

                    # Extract statistics
                    for key,val in batch_metrics.items():
                        if key != "loss":
                            if self.sliding_window and not first_batch:
                                metrics[key] = np.concatenate( (metrics.get(key,[]), [val[-1]]), axis=None )
                                metrics[key + "_sum"] = metrics.get( key + "_sum",0) + val[-1]
                            else:
                                metrics[key] = np.concatenate( (metrics.get(key,[]), val), axis=None )
                                metrics[key + "_sum"] = metrics.get( key + "_sum",0) + np.sum(val)

                    if self.sliding_window and not first_batch:
                        loss = self.loss_(output[np.newaxis,-1,:], [target[-1]]).mean(axis=1).sum()
                        example_cnt += 1
                        pbar.update(1)
                    else:
                        loss = self.loss_(output, target).mean(axis=1).sum()
                        example_cnt += data.shape[0]
                        pbar.update(data.shape[0])

                    metrics["loss"] = np.concatenate( (metrics.get("loss",[]), loss), axis=None )
                    metrics["loss_sum"] = metrics.get( "loss_sum",0) + np.sum(loss)

                    metrics["time"] = np.concatenate( (metrics.get("time",[]), batch_time / data.shape[0]), axis=None )
                    metrics["time_sum"] = metrics.get( "time_sum",0) + np.sum(batch_time / data.shape[0])

                    m_str = ""
                    for key,val in metrics.items():
                        if "_sum" in key:
                            m_str += "{} {:2.4f} ".format(key.split("_sum")[0], val / example_cnt)
                    
                    desc = '[{}/{}] {}'.format(
                        epoch, 
                        self.epochs-1, 
                        m_str
                    )
                    pbar.set_description(desc)

                    first_batch = False
                
                if self.eval_every_epochs is not None and epoch % self.eval_every_epochs == 0 and self.out_path is not None:
                    np.save(os.path.join(self.out_path, "epoch_{}.npy".format(epoch)), metrics, allow_pickle=True)