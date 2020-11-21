import numpy as np
import random
from tqdm import tqdm

from DecisionTree import generate_new_tree, tensor_predict_proba

from joblib import Parallel, delayed

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from sklearn.tree import _tree

from plotille import histogram

class BiasedProxEnsemble:
    def __init__(self,  
                optimizer_cfg, 
                max_depth = 5,
                mode = "random",
                weights_per_class = False,
                seed = None,
                verbose = True, 
                out_path = None, 
                x_test = None, 
                y_test = None, 
                eval_every = 5,
                store_on_eval = False,
                n_jobs = 8) :
        
        self.optimizer_cfg = optimizer_cfg
        self.weights_per_class = weights_per_class
        self.seed = seed
        self.verbose = verbose
        self.out_path = out_path
        self.x_test = x_test
        self.y_test = y_test
        self.eval_every = eval_every
        self.store_on_eval = store_on_eval
        self.max_depth = max_depth 
        self.n_jobs = n_jobs
        self.mode = mode

        if self.seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def predict_proba(self, X, eval_mode=True):
        # TODO
        # if self.pipeline:
        #     ret_val = apply_in_batches(self, self.pipeline.transform(X), batch_size=self.batch_size)
        # else:
        #     ret_val = apply_in_batches(self, X, batch_size=self.batch_size)
        #X = self._validate_X_predict(X)

        def single_predict_proba(h,w,X):
            #return w*h.predict_proba(X)
            return h.predict_proba(X)

        all_proba = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(single_predict_proba) (h,w,X) for h,w in zip(self.estimators_, self.estimator_weights_)
        )
        all_proba = np.array(all_proba)
        scaled_prob = np.array([w*p for w,p in zip(all_proba, self.estimator_weights_)])
        combined_proba = np.sum(scaled_prob, axis=0)
        # if len(all_proba) == 1:
        #     return combined_proba[0],all_proba
        # else:
        return combined_proba,all_proba

    # Modified from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    def create_mini_batches(self, inputs, targets, batch_size, shuffle=False):
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

    def fit(self, X, y, sample_weight = None):
        self.X_ = X
        self.y_ = y
        
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_

        if self.out_path is not None:
            outfile = open(self.out_path + "/training.csv", "w", 1)
            if self.x_test is not None:
                outfile.write("epoch,train-loss,train-accuracy,nonzero,test-loss,test-accuracy\n")
            else:
                outfile.write("epoch,train-loss,train-accuracy,nonzero\n")
        
        self.estimator_weights_ = []
        self.estimators_ = []

        epochs = self.optimizer_cfg.get("epochs",10)
        batch_size = self.optimizer_cfg.get("batch_size", 32)
        alpha = self.optimizer_cfg.get("alpha", 1e-3)
        l_reg = self.optimizer_cfg.get("lambda",0)
        loss_function = self.optimizer_cfg.get("loss_function", None)
        loss_function_deriv = self.optimizer_cfg.get("loss_function_deriv", None)

        assert loss_function is not None and loss_function_deriv is not None, "BiasedProxEnsemble: loss_function and loss_function_deriv must not be None!"

        for epoch in range(epochs):
            mini_batches = self.create_mini_batches(self.X_, self.y_, batch_size, True) 
            epoch_loss = 0
            batch_cnt = 0
            avg_accuarcy = 0
            epoch_nonzero = 0

            with tqdm(total=X.shape[0], ncols=135, disable = not self.verbose) as pbar:
                for batch in mini_batches: 
                    data, target = batch 

                    # TODO Test 1.0 initialization
                    if self.weights_per_class:
                        self.estimator_weights_.append([0.0 for i in range(self.n_outputs_)]) 
                    else:
                        self.estimator_weights_.append(0.0)

                    #self.estimators_.append(TensorTree(self.mode, data, target, self.max_depth, data.shape[1], self.n_classes_, None))
                    self.estimators_.append(generate_new_tree(self.mode, data, target, self.max_depth, data.shape[1], self.n_classes_, None))

                    target_one_hot = np.array([ [1 if t == c else 0 for c in self.classes_] for t in target])
                    #output, all_proba = self.predict_proba(data)
                    output, all_proba = tensor_predict_proba(data)
                    
                    loss = loss_function(output, target_one_hot)
                    loss_deriv = loss_function_deriv(output, target_one_hot)
                    
                    epoch_loss += np.mean(loss)
                    accuracy = accuracy_score(target, output.argmax(axis=1))*100.0
                    avg_accuarcy += accuracy
                    batch_cnt += 1
                    
                    if self.weights_per_class:
                        directions = np.mean(all_proba*loss_deriv,axis=1)
                    else:
                        directions = np.mean(all_proba*loss_deriv,axis=(1,2))
                    
                    tmp_w = self.estimator_weights_ - alpha*directions
                    sign = np.sign(tmp_w)
                    tmp_w = np.abs(tmp_w)-l_reg
                    
                    self.estimator_weights_ = sign*np.maximum(tmp_w,0)
                    self.estimators_ = [e for i, e in enumerate(self.estimators_) if self.estimator_weights_[i] != 0]
                    self.estimator_weights_ = [w for w in self.estimator_weights_ if w != 0]
                    epoch_nonzero += len(self.estimators_)

                    pbar.update(data.shape[0])
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f} nonzero {:2.4f}'.format(
                        epoch, 
                        epochs-1, 
                        epoch_loss/batch_cnt, 
                        avg_accuarcy/batch_cnt,
                        epoch_nonzero/batch_cnt
                    )
                    pbar.set_description(desc)
                
                if self.out_path is not None and (self.x_test is None or self.y_test is None):
                    outfile.write("{},{},{},{}\n".format(epoch,epoch_loss/batch_cnt,avg_accuarcy/batch_cnt,np.count_nonzero(self.estimator_weights_)))
                else:
                    output,_ = self.predict_proba(self.x_test)
                    target_one_hot = np.array([ [1 if t == c else 0 for c in self.classes_] for t in self.y_test])

                    test_loss = np.mean(loss_function(output, target_one_hot))
                    test_accuracy = accuracy_score(self.y_test, self.proba_to_class(output))*100.0
                    outfile.write("{},{},{},{},{},{}\n".format(epoch,epoch_loss/batch_cnt,avg_accuarcy/batch_cnt,np.count_nonzero(self.estimator_weights_),test_loss,test_accuracy))

                    desc = '[{}/{}] loss {:2.4f} train-acc {:2.4f} nonzero {:2.4f} test-loss {:2.4f} test-acc {:2.4f}'.format(
                        epoch, 
                        epochs-1, 
                        epoch_loss/batch_cnt, 
                        avg_accuarcy/batch_cnt,
                        len(self.estimators_),
                        test_loss,
                        test_accuracy
                    )
                    pbar.set_description(desc)
            # print(histogram(self.weights.flatten(), bins=20, height=20))
            
        