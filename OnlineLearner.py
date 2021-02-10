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
                x_test = None, 
                y_test = None, 
                out_file = None,
                eval_every_items = None,
                eval_every_epochs = None):
        
        assert eval_loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert epochs >= 1, "epochs must be at-least 1"

        self.eval_loss = eval_loss
        self.batch_size = batch_size
        self.sliding_window = sliding_window
        self.epochs = epochs
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
        # self.X_ = X
        # self.y_ = y 
        
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
            mini_batches = create_mini_batches(X, y, self.batch_size, self.shuffle,self.sliding_window) 
            epoch_loss = 0
            batch_cnt = 0
            sum_accuracy = 0
            n_trees = 0
            n_params = 0
            last_stored = 0
            epoch_time = 0

            new_epoch = epoch > 0 
            first_batch = True

            with tqdm(total=X.shape[0], ncols=135, disable = not self.verbose) as pbar:
                for batch in mini_batches: 
                    data, target = batch 
                    
                    # Compute current statistics
                    if self.sliding_window and not first_batch:
                        output = self.predict_proba(data[-1].reshape(1,data.shape[1]))
                        loss = self.loss_(output, [target[-1]]).mean()
                        accuracy = accuracy_score([target[-1]], output.argmax(axis=1))*100.0
                        cur_batch_size = 1
                    else:
                        output = self.predict_proba(data)
                        loss = self.loss_(output, target).mean()
                        accuracy = accuracy_score(target, output.argmax(axis=1))*100.0
                        cur_batch_size = data.shape[0]
                    num_trees = self.num_trees() 
                    num_params = self.num_parameters() 

                    # Update Model                    
                    start_time = time.time()
                    _, _, updates = self.next(data, target, train = True, new_epoch = new_epoch)
                    batch_time = time.time() - start_time

                    # Update running statistics
                    epoch_time += 1000 * batch_time / cur_batch_size
                    first_batch = False
                    total_model_updates += updates
                    n_trees += num_trees
                    n_params += num_params
                    epoch_loss += loss
                    sum_accuracy += accuracy
                    batch_cnt += 1
                    total_item_cnt += cur_batch_size
                    last_stored += cur_batch_size

                    if self.sliding_window:
                        pbar.update(1)
                    else:
                        pbar.update(data.shape[0])

                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f} n_trees {:2.4f} n_params {:2.4f} time_item {:2.4f}'.format(
                        epoch, 
                        epochs-1, 
                        epoch_loss/batch_cnt, 
                        sum_accuracy/batch_cnt,
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
                        
                        out_dict["item_loss"] = loss
                        out_dict["item_accuracy"] = accuracy 
                        out_dict["item_num_trees"] = num_trees
                        out_dict["item_num_parameters"] = num_params
                        out_dict["item_time"] = batch_time / cur_batch_size
                        out_dict["total_model_updates"] = total_model_updates

                        out_dict["train_loss"] = epoch_loss/batch_cnt
                        out_dict["train_accuracy"] = sum_accuracy/batch_cnt
                        out_dict["train_num_trees"] = n_trees/batch_cnt
                        out_dict["train_num_parameters"] = n_params/batch_cnt
                        out_dict["total_item_cnt"] = total_item_cnt
                        out_dict["epoch"] = epoch
                        out_str = json.dumps(out_dict)
                        fout.write(out_str + "\n")
                        last_stored = 0
                
                if self.eval_every_epochs is not None and self.eval_every_epochs > 0 and epoch % self.eval_every_epochs == 0:
                    out_dict = {}
                    if self.x_test is not None and self.y_test is not None:
                        tmp_dict = self.eval(self.x_test, self.y_test)
                        for key, val in tmp_dict.items():
                            out_dict["test_" + key] = val
                    
                    # out_dict["item_loss"] = loss
                    # out_dict["item_accuracy"] = accuracy 
                    # out_dict["item_num_trees"] = num_trees 
                    # out_dict["item_num_parameters"] = num_params 
                    # out_dict["item_time"] = batch_time / data.shape[0]
                    out_dict["total_model_updates"] = total_model_updates

                    out_dict["train_loss"] = epoch_loss/batch_cnt
                    out_dict["train_accuracy"] = sum_accuracy/batch_cnt
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
                            sum_accuracy/batch_cnt,
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
                            sum_accuracy/batch_cnt,
                            n_trees/batch_cnt,
                            n_params/batch_cnt,
                            epoch_time / batch_cnt
                        )
                    pbar.set_description(desc)
