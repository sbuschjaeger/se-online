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
                epochs = None,
                n_updates = None,
                seed = None,
                verbose = True, 
                shuffle = True,
                x_test = None, 
                y_test = None, 
                out_file = None,
                eval_every_items = None,
                eval_every_epochs = None):
        
        assert n_updates is None or n_updates >= 1, "n_updates must be either None or a >= 1"
        assert epochs is None or epochs >= 1, "epochs must be either None or a >= 1"
        assert epochs is not None or n_updates is not None, "n_updates and epochs cannot both be None"

        self.batch_size = batch_size
        self.epochs = epochs
        self.n_updates = n_updates
        self.verbose = verbose
        self.shuffle = shuffle
        self.x_test = x_test
        self.y_test = y_test
        self.eval_every_items = eval_every_items
        self.eval_every_epochs = eval_every_epochs
        self.out_file = out_file

        if seed is None:
            self.seed = 1234
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.n_updates is not None and self.epochs is not None:
            print("WARNING: n_updates and epochs are both not None. Please pick one. I am picking epochs for you now")
            self.n_updates = None

    @abstractmethod
    def next(self, data, target, train=False, new_epoch = False):
        pass

    def eval(self, X, y):
        test_batches = create_mini_batches(X, y, self.batch_size, True) 
        test_accuracy = 0
        test_loss = 0
        test_n_trees = 0
        test_n_parameters = 0
        
        test_cnt = 0
        for batch in test_batches: 
            data, target = batch 
            metrics, output, _ = self.next(data, target, train = False)

            test_loss += metrics["loss"]
            test_n_trees += metrics["num_trees"] 
            test_n_parameters += metrics["num_parameters"] 
            test_accuracy += accuracy_score(target, output.argmax(axis=1))*100.0
            test_cnt += 1
        return {"loss":test_loss / test_cnt, "num_trees":test_n_trees / test_cnt, "accuracy": test_accuracy / test_cnt, "num_parameters" : test_n_parameters / test_cnt}

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

        if self.epochs is None:
            self.epochs = int(max(self.n_updates / (X.shape[0] / self.batch_size), 1)) + 1

        epochs = self.epochs
        total_item_cnt = 0
        total_model_updates = 0
        for epoch in range(self.epochs):
            mini_batches = create_mini_batches(self.X_, self.y_, self.batch_size, self.shuffle) 
            epoch_loss = 0
            batch_cnt = 0
            avg_accuarcy = 0
            n_trees = 0
            n_params = 0
            last_stored = 0
            epoch_time = 0

            first_batch = epoch > 0 
            if self.n_updates is not None:
                tqdm_total = int(min(self.n_updates - total_model_updates, X.shape[0] / self.batch_size))
            else:
                tqdm_total = X.shape[0]
            #print("tqdm_total:", tqdm_total)

            with tqdm(total=tqdm_total, ncols=135, disable = not self.verbose) as pbar:
                for batch in mini_batches: 
                    if self.n_updates is not None and self.n_updates < total_model_updates:
                        break
                    data, target = batch 

                    start_time = time.time()
                    metrics, output, updates = self.next(data, target, train = True, new_epoch = first_batch)
                    batch_time = time.time() - start_time
                    epoch_time += 1000 * batch_time / data.shape[0]
                    first_batch = False

                    total_model_updates += updates
                    epoch_loss += metrics["loss"]
                    n_trees += metrics["num_trees"] 
                    n_params += metrics["num_parameters"] 

                    accuracy = accuracy_score(target, output.argmax(axis=1))*100.0
                    avg_accuarcy += accuracy
                    batch_cnt += 1
                    total_item_cnt += data.shape[0]
                    last_stored += data.shape[0]

                    if self.n_updates is not None:
                        pbar.update(updates)
                    else:
                        pbar.update(data.shape[0])

                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f} n_trees {:2.4f} n_params {:2.4f} time_item {:2.4f}'.format(
                        epoch, 
                        epochs-1, 
                        epoch_loss/batch_cnt, 
                        avg_accuarcy/batch_cnt,
                        n_trees/batch_cnt,
                        n_params/batch_cnt,
                        epoch_time / batch_cnt
                    )
                    pbar.set_description(desc)
                    if self.eval_every_items is not None and last_stored >= self.eval_every_items and self.eval_every_items > 0:
                        out_dict = {}
                        if self.x_test is not None and self.y_test is not None:
                            tmp_dict = self.eval(self.x_test, self.y_test)
                            for key, val in tmp_dict.items():
                                out_dict["test_" + key] = val
                        
                        out_dict["item_loss"] = metrics["loss"] 
                        out_dict["item_accuracy"] = accuracy 
                        out_dict["item_num_trees"] = metrics["num_trees"] 
                        out_dict["item_num_parameters"] = metrics["num_parameters"] 
                        out_dict["item_time"] = batch_time / data.shape[0]
                        out_dict["total_model_updates"] = total_model_updates

                        out_dict["train_loss"] = epoch_loss/batch_cnt
                        out_dict["train_accuracy"] = avg_accuarcy/batch_cnt
                        out_dict["train_num_trees"] = n_trees/batch_cnt
                        out_dict["train_num_parameters"] = n_params/batch_cnt
                        out_dict["total_item_cnt"] = total_item_cnt
                        out_dict["epoch"] = epoch
                        out_str = json.dumps(out_dict)
                        fout.write(out_str + "\n")
                        last_stored = 0
                
                if self.n_updates is not None and self.n_updates < total_model_updates:
                    break

                if self.eval_every_epochs is not None and self.eval_every_epochs > 0 and epoch % self.eval_every_epochs == 0:
                    out_dict = {}
                    if self.x_test is not None and self.y_test is not None:
                        tmp_dict = self.eval(self.x_test, self.y_test)
                        for key, val in tmp_dict.items():
                            out_dict["test_" + key] = val
                    
                    out_dict["item_loss"] = metrics["loss"] 
                    out_dict["item_accuracy"] = accuracy 
                    out_dict["item_num_trees"] = metrics["num_trees"] 
                    out_dict["item_num_parameters"] = metrics["num_parameters"] 
                    out_dict["item_time"] = batch_time / data.shape[0]
                    out_dict["total_model_updates"] = total_model_updates

                    out_dict["train_loss"] = epoch_loss/batch_cnt
                    out_dict["train_accuracy"] = avg_accuarcy/batch_cnt
                    out_dict["train_num_trees"] = n_trees/batch_cnt
                    out_dict["train_num_parameters"] = n_params/batch_cnt
                    out_dict["total_item_cnt"] = total_item_cnt
                    out_dict["epoch"] = epoch
                    out_str = json.dumps(out_dict)
                    fout.write(out_str + "\n")

                    if self.x_test is not None and self.y_test is not None:
                        desc = '[{}/{}] loss {:2.4f} acc {:2.4f} n_trees {:2.4f} n_params {:2.4f} time_item {:2.4f} test-acc {:2.4f}'.format(
                            epoch, 
                            epochs-1, 
                            epoch_loss/batch_cnt, 
                            avg_accuarcy/batch_cnt,
                            n_trees/batch_cnt,
                            n_params/batch_cnt,
                            epoch_time / batch_cnt,
                            out_dict["test_accuracy"]
                        )
                    else:
                         desc = '[{}/{}] loss {:2.4f} acc {:2.4f} n_trees {:2.4f} n_params {:2.4f} time_item {:2.4f}'.format(
                            epoch, 
                            epochs-1, 
                            epoch_loss/batch_cnt, 
                            avg_accuarcy/batch_cnt,
                            n_trees/batch_cnt,
                            n_params/batch_cnt,
                            epoch_time / batch_cnt
                        )
                    pbar.set_description(desc)
