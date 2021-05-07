import os
from types import MethodWrapperType
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
                seed = None,
                verbose = True, 
                shuffle = True,
                out_path = None):
        
        assert eval_loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"

        self.eval_loss = eval_loss
        self.verbose = verbose
        self.shuffle = shuffle
        self.out_path = out_path

        if seed is None:
            self.seed = 1234
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)

    @abstractmethod
    def next(self, data, target):
        pass
    
    @abstractmethod
    def num_trees(self):
        pass
    
    @abstractmethod
    def num_parameters(self):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def compute_loss(self, output, target):
        target_one_hot = np.array( [1.0 if target == i else 0.0 for i in range(self.n_classes_)] )
        if self.eval_loss == "mse":
            loss = (output - target_one_hot) * (output - target_one_hot)
        elif self.eval_loss == "cross-entropy":
            p = softmax(output)
            loss = -target_one_hot*np.log(p + 1e-7)
        elif self.eval_loss == "hinge2":
            zeros = np.zeros_like(target_one_hot)
            loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
        else:
            raise "Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.eval_loss)
        return loss.mean() # For multiclass problems we use the mean over all classes

    def compute_metrics(self, C, n):
        p0 = np.trace(C) / n
        pc = 0
        for i in range(len(C)):
            pc += np.sum(C[i,:]) / n * np.sum(C[:,i]) / n
        pc /= n

        if (p0 - pc) == 0:
            kappa = 0
        else:
            kappa = (p0 - pc) / (1.0 - pc)

        accuracy = p0 * 100.0

        f1 = 0
        for i in range(self.n_classes_):
            tp = np.sum(C[i,i])
            fp = np.sum(C[i, np.concatenate((np.arange(0, i), np.arange(i+1, self.n_classes_)))])
            fn = np.sum(C[np.concatenate((np.arange(0, i), np.arange(i+1, self.n_classes_))), i])

            precision = tp/(tp+fp) if (tp+fp) > 0 else 0
            recall = tp/(tp+fn) if (tp+fn) > 0 else 0
            # precision = C[i][i] 
            # recall = np.sum(C[:,i])

            if precision != 0 and recall != 0:
                f1 += 2.0 * precision * recall / (precision + recall)
        f1 /= self.n_classes_

        return {"kappa":kappa, "accuracy":accuracy, "f1":f1}
    # def loss_(self, output, target):
    #     if self.eval_loss == "mse":
    #         target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
    #         loss = (output - target_one_hot) * (output - target_one_hot)
    #     elif self.eval_loss == "cross-entropy":
    #         target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
    #         p = softmax(output, axis=1)
    #         loss = -target_one_hot*np.log(p + 1e-7)
    #     elif self.eval_loss == "hinge2":
    #         target_one_hot = np.array( [ [1.0 if y == i else -1.0 for i in range(self.n_classes_)] for y in target] )
    #         zeros = np.zeros_like(target_one_hot)
    #         loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
    #     else:
    #         raise "Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.eval_loss)
    #     return loss

    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_
        
        self.X_ = X
        self.y_ = y

        metrics = {}
        C = np.zeros( (self.n_classes_, self.n_classes_) )
        n = 0
        with tqdm(total=X.shape[0], ncols=150, disable = not self.verbose) as pbar:
            for x,y in zip(X,y):
                output = self.predict_proba(x)

                # Update Model                    
                start_time = time.time()
                self.next(x, y)
                item_time = time.time() - start_time
                
                n += 1
                ypred = output.argmax()
                C[ypred][y] += 1
                item_metrics = self.compute_metrics(C, n)
                item_metrics["time"] = item_time
                item_metrics["loss"] = self.compute_loss(output, y)
                
                # Extract statistics and also compute cumulative sum for plotting later
                for key,val in item_metrics.items():
                    metrics[key] = np.concatenate( (metrics.get(key,[]), val), axis=None )
                    if key + "_sum" in metrics:
                        metrics[key + "_sum"].append(metrics[key + "_sum"][-1] + val)
                    else:
                        metrics[key + "_sum"] = [val]

                metrics["item_cnt"] = np.concatenate( (metrics.get("item_cnt",[]), n), axis=None )
                pbar.update(1)

                m_str = ""
                for key,val in metrics.items():
                    if "_sum" in key:
                        m_str += "{} {:2.4f} ".format(key.split("_sum")[0], val[-1] / n)
                
                desc = '{}'.format(
                    m_str
                )
                pbar.set_description(desc)

                # if item_cnt > 1500:
                #     break

            if self.out_path is not None:
                # TODO add gzip here
                np.save(os.path.join(self.out_path, "training.npy"), metrics, allow_pickle=True)

        # for epoch in range(self.epochs):
        #     mini_batches = create_mini_batches(X, y, self.batch_size, self.shuffle, self.sliding_window) 

        #     metrics = {}

        #     new_epoch = epoch > 0 
        #     first_batch = True
        #     example_cnt = 0

        #     with tqdm(total=X.shape[0], ncols=150, disable = not self.verbose) as pbar:
        #         for batch in mini_batches: 
        #             data, target = batch 
                    
        #             # Update Model                    
        #             start_time = time.time()
                    
        #             # TODO remove batch size from this class. Next should only accept one item / target
        #             batch_metrics, output = self.next(data, target, train = True, new_epoch = new_epoch)
        #             batch_time = time.time() - start_time

        #             # Extract statistics
        #             for key,val in batch_metrics.items():
        #                 if key != "loss":
        #                     if self.sliding_window and not first_batch:
        #                         metrics[key] = np.concatenate( (metrics.get(key,[]), [val[-1]]), axis=None )
        #                         metrics[key + "_sum"] = metrics.get( key + "_sum",0) + val[-1]
        #                     else:
        #                         metrics[key] = np.concatenate( (metrics.get(key,[]), val), axis=None )
        #                         metrics[key + "_sum"] = metrics.get( key + "_sum",0) + np.sum(val)

        #             if self.sliding_window and not first_batch:
        #                 loss = self.loss_(output[np.newaxis,-1,:], [target[-1]]).mean(axis=1).sum()
        #                 example_cnt += 1
        #                 pbar.update(1)
        #             else:
        #                 loss = self.loss_(output, target).mean(axis=1).sum()
        #                 example_cnt += data.shape[0]
        #                 pbar.update(data.shape[0])

        #             metrics["loss"] = np.concatenate( (metrics.get("loss",[]), loss), axis=None )
        #             metrics["loss_sum"] = metrics.get( "loss_sum",0) + np.sum(loss)

        #             metrics["time"] = np.concatenate( (metrics.get("time",[]), batch_time / data.shape[0]), axis=None )
        #             metrics["time_sum"] = metrics.get( "time_sum",0) + np.sum(batch_time / data.shape[0])

        #             metrics["item_cnt"] = np.concatenate( (metrics.get("item_cnt",[]), example_cnt), axis=None )

        #             m_str = ""
        #             for key,val in metrics.items():
        #                 if "_sum" in key:
        #                     m_str += "{} {:2.4f} ".format(key.split("_sum")[0], val / example_cnt)
                    
        #             desc = '[{}/{}] {}'.format(
        #                 epoch, 
        #                 self.epochs-1, 
        #                 m_str
        #             )
        #             pbar.set_description(desc)

        #             first_batch = False

        #         if self.eval_every_epochs is not None and epoch % self.eval_every_epochs == 0 and self.out_path is not None:
        #             np.save(os.path.join(self.out_path, "epoch_{}.npy".format(epoch)), metrics, allow_pickle=True)