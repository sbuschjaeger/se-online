import numpy as np
import random
from tqdm import tqdm
import json

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
def create_mini_batches(inputs, targets, batch_size, shuffle=False):
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
        start_idx += batch_size
        yield inputs[excerpt], targets[excerpt]

class OnlineLearner(ABC):
    def __init__(self,  
                batch_size = 256,
                epochs = 50,
                seed = None,
                verbose = True, 
                x_test = None, 
                y_test = None, 
                out_file = None,
                eval_every_items = None,
                eval_every_epochs = None):
                        
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.x_test = x_test
        self.y_test = y_test
        self.eval_every_items = eval_every_items
        self.eval_every_epochs = eval_every_epochs
        self.out_file = out_file

        if seed is None:
            self.seed = 1234
        else:
            self.seed= seed

        np.random.seed(self.seed)
        random.seed(self.seed)

    @abstractmethod
    def next(self, data, target, train=False, new_epoch = False):
        pass

    def eval(self, X, y):
        test_batches = create_mini_batches(X, y, self.batch_size, True) 
        test_accuracy = 0
        test_loss = 0
        test_n_trees = 0
        test_n_nodes = 0
        
        test_cnt = 0
        for batch in test_batches: 
            data, target = batch 
            metrics, output = self.next(data, target, train = False)

            test_loss += metrics["loss"]
            test_n_trees += metrics["num_trees"] 
            test_n_nodes += metrics["num_nodes"] 
            test_accuracy += accuracy_score(target, output.argmax(axis=1))*100.0
            test_cnt += 1
        return {"test_loss":test_loss / test_cnt, "test_num_trees":test_n_trees / test_cnt, "test_accuracy": test_accuracy / test_cnt, "test_num_nodes" : test_n_nodes / test_cnt}

    def fit(self, X, y, sample_weight = None):
        self.X_ = X
        self.y_ = y
        
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_
        
        if self.out_file is not None:
            fout = open(self.out_file, "w")
        else:
            fout = None

        epochs = self.epochs
        total_item_cnt = 0
        for epoch in range(epochs):
            mini_batches = create_mini_batches(self.X_, self.y_, self.batch_size, True) 
            epoch_loss = 0
            batch_cnt = 0
            avg_accuarcy = 0
            n_trees = 0
            n_nodes = 0
            last_stored = 0

            with tqdm(total=X.shape[0], ncols=135, disable = not self.verbose) as pbar:
                first_batch = True
                for batch in mini_batches: 
                    data, target = batch 
                    metrics, output = self.next(data, target, train = True, new_epoch = first_batch)
                    first_batch = False

                    epoch_loss += metrics["loss"]
                    n_trees += metrics["num_trees"] 
                    n_nodes += metrics["num_nodes"] 

                    accuracy = accuracy_score(target, output.argmax(axis=1))*100.0
                    avg_accuarcy += accuracy
                    batch_cnt += 1
                    total_item_cnt += data.shape[0]
                    last_stored += data.shape[0]

                    pbar.update(data.shape[0])
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f} n_trees {:2.4f} n_nodes {:2.4f}'.format(
                        epoch, 
                        epochs-1, 
                        epoch_loss/batch_cnt, 
                        avg_accuarcy/batch_cnt,
                        n_trees/batch_cnt,
                        n_nodes/batch_cnt
                    )
                    pbar.set_description(desc)

                    if all(v is not None for v in [self.x_test, self.y_test, self.eval_every_items]) and self.eval_every_items > 0 and last_stored > self.eval_every_items:
                        out_dict = self.eval(self.x_test, self.y_test)
                        out_dict["total_item_cnt"] = total_item_cnt
                        out_dict["epoch"] = epoch
                        out_str = json.dumps(out_dict)
                        fout.write(out_str + "\n")
                        last_stored = 0

                if all(v is not None for v in [self.x_test, self.y_test, self.eval_every_items]) and self.eval_every_epochs > 0 and epoch % self.eval_every_epochs == 0:
                    out_dict = self.eval(self.x_test, self.y_test)
                    out_dict["total_item_cnt"] = total_item_cnt
                    out_dict["epoch"] = epoch
                    out_str = json.dumps(out_dict)
                    fout.write(out_str + "\n")

                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f} n_trees {:2.4f} n_nodes {:2.4f} test-acc {:2.4f}'.format(
                        epoch, 
                        epochs-1, 
                        epoch_loss/batch_cnt, 
                        avg_accuarcy/batch_cnt,
                        n_trees/batch_cnt,
                        n_nodes/batch_cnt,
                        out_dict["test_accuracy"]
                    )
                    pbar.set_description(desc)
