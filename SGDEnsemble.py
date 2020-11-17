import numpy as np
import random
from tqdm import tqdm

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

def get_rounded(dts):
    # TODO THIS MAKES ONLY SENSE FOR BINARY CLASSIFICATION PROBLEMS!!!
    def round_pred(tree, cur_node = 0):
        if tree.children_left[cur_node] == _tree.TREE_LEAF and tree.children_right[cur_node] == _tree.TREE_LEAF:
            #tree.value[cur_node][0] = [0, 0]
            if tree.value[cur_node][0][0] > tree.value[cur_node][0][1]:
                tree.value[cur_node][0] = [1, 0]
            else:
                tree.value[cur_node][0] = [0, 1]
        else:
            round_pred(tree, tree.children_left[cur_node])
            round_pred(tree, tree.children_right[cur_node])

    rounded_dts = []
    for dt in dts:
        new_dt = deepcopy(dt)
        round_pred(new_dt.tree_)
        rounded_dts.append(new_dt)
    return rounded_dts

def get_complement(dts):
    # TODO THIS MAKES ONLY SENSE FOR BINARY CLASSIFICATION PROBLEMS!!!
    def swap_pred(tree, cur_node = 0):
        if tree.children_left[cur_node] == _tree.TREE_LEAF and tree.children_right[cur_node] == _tree.TREE_LEAF:
            tree.value[cur_node][0] = [tree.value[cur_node][0][1],tree.value[cur_node][0][0]]
        else:
            swap_pred(tree, tree.children_left[cur_node])
            swap_pred(tree, tree.children_right[cur_node])

    complement_dts = []
    for dt in dts:
        new_dt = deepcopy(dt)
        swap_pred(new_dt.tree_)
        complement_dts.append(new_dt)
    return complement_dts

class SGDEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self,  
                 forest_options,
                 optimizer,
                 loss_function,
                 loss_function_deriv,
                 weights_per_class = False,
                 # TODO: This is deprecated
                 use_complement_classifier = False,
                 # TODO: This is deprecated
                 use_proba = False,
                 n_jobs = 8,
                 pipeline = None,
                 seed = None,
                 verbose = True, 
                 out_path = None, 
                 x_test = None, 
                 y_test = None, 
                 eval_test = 5,
                 store_on_eval = False) :
        super().__init__()
        
        self.forest_options = forest_options
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loss_function_deriv = loss_function_deriv
        self.weights_per_class = weights_per_class
        self.use_complement_classifier = use_complement_classifier
        self.use_proba = use_proba

        self.pipeline = pipeline
        self.verbose = verbose
        self.out_path = out_path
        self.x_test = x_test
        self.y_test = y_test
        self.seed = seed
        self.eval_test = eval_test
        self.store_on_eval = store_on_eval
        self.n_jobs = n_jobs

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def store(self, out_path, dim, name="model"):
        raise NotImplementedError()
    
    # Stolen directly from ForestClassifier -> predict in scikit-learn/sklearn/ensemble/forest.py (line 517-)
    def proba_to_class(self, proba):
         # self.n_classes_ was self.n_outputs
        
        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            predictions = np.zeros((n_samples, self.n_outputs_))

            for k in range(self.n_classes_):
                predictions[:, k] = self.n_outputs_[k].take(np.argmax(proba[k],axis=1),axis=0)

            return predictions
    
    def predict(self, X, eval_mode=True):
        proba,_ = self.predict_proba(X)
        return self.proba_to_class(proba)

    # Stolen directly from ForestClassifier -> _validate_X_predict in scikit-learn/sklearn/ensemble/forest.py (line 350-)
    def _validate_X_predict(self, X):
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    def predict_proba(self, X, eval_mode=True):
        # TODO
        # if self.pipeline:
        #     ret_val = apply_in_batches(self, self.pipeline.transform(X), batch_size=self.batch_size)
        # else:
        #     ret_val = apply_in_batches(self, X, batch_size=self.batch_size)
        X = self._validate_X_predict(X)

        def single_predict_proba(h,w,X):
            #return w*h.predict_proba(X)
            return h.predict_proba(X)

        all_proba = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(single_predict_proba) (h,w,X) for h,w in zip(self.estimators_, self.estimator_weights_)
        )
        all_proba = np.array(all_proba)
        scaled_prob = np.array([w*p for w,p in zip(all_proba, self.estimator_weights_)])
        combined_proba = np.sum(scaled_prob, axis=0)
        
        if len(all_proba) == 1:
            return combined_proba[0],all_proba
        else:
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
        # self.classes_ = unique_labels(y)
        # self.n_classes_ = len(self.classes_)
        if self.pipeline:
            X = self.pipeline.fit_transform(X)

        self.X_ = X
        self.y_ = y
        
        print("Preparing set of models")
        forest_classifier = self.forest_options.pop("model", ExtraTreesClassifier)
        # print(forest_classifier)
        # asdf
        # if (isinstance(model, (RandomForestClassifier,ExtraTreesClassifier))):
        #     self.forest_options["max_samples"] = self.optimizer["batch_size"]
        #     self.forest_options["bootstrap"] = True
        
        # TODO: This is currently pure laziness to set n_outputs / classes_ correctly
        forest = forest_classifier(**self.forest_options)
        forest.fit(self.X_, self.y_)
        self.n_outputs_ = forest.estimators_[0].n_outputs_
        self.classes_ = forest.classes_
        self.n_classes_ = forest.n_classes_

        if self.out_path is not None:
            outfile = open(self.out_path + "/training.csv", "w", 1)
            if self.x_test is not None:
                outfile.write("epoch,train-loss,train-accuracy,nonzero,test-loss,test-accuracy\n")
            else:
                outfile.write("epoch,train-loss,train-accuracy,nonzero\n")

        train_output = forest.predict_proba(X)
        train_y = np.array([ [1 if t == c else 0 for c in self.classes_] for t in y])
        train_loss = np.mean(self.loss_function(train_output, train_y))
        train_accuracy = accuracy_score(y, self.proba_to_class(train_output))*100.0

        if hasattr(forest, "estimator_weights_"):
            nonzero = np.count_nonzero(forest.estimator_weights_)
        else:
            nonzero = len(forest.estimators_)

        if self.x_test is not None and self.y_test is not None:
            test_output = forest.predict_proba(self.x_test)
            test_y = np.array([ [1 if t == c else 0 for c in self.classes_] for t in self.y_test])
            test_loss = np.mean(self.loss_function(test_output, test_y))
            test_accuracy = accuracy_score(self.y_test, self.proba_to_class(test_output))*100.0
            outfile.write("{},{},{},{},{},{}\n".format(-1,train_loss,train_accuracy,nonzero,test_loss,test_accuracy))
        else:
            outfile.write("{},{},{},{}\n".format(-1,train_loss,train_accuracy,nonzero))
        
        if not self.use_proba:
            base_models = get_rounded(forest.estimators_)
        else:
            base_models = forest.estimators_

        if self.use_complement_classifier:
            complement = get_complement(base_models)
            self.estimators_ = base_models + complement
        else:
            self.estimators_ = base_models
        
        if self.weights_per_class:
            self.estimator_weights_ = np.zeros((len(self.estimators_), self.n_classes_))
            #self.weights = 1.0 / len(self.estimators_) * np.ones((len(self.estimators_), self.n_classes_))
            #self.weights = np.random.normal(0, 0.01, (len(self.estimators_), self.n_classes_))
        else:
            self.estimator_weights_ = np.zeros((len(self.estimators_), ))
            #self.weights = np.random.normal(0, 0.01, (len(self.estimators_),))
            #self.weights = 1.0 / len(self.estimators_) * np.ones((len(self.estimators_), ))
        #self.weights = forest.estimator_weights_

        epochs = self.optimizer.get("epochs",10)
        batch_size = self.optimizer.get("batch_size", 32)
        alpha = self.optimizer.get("alpha", 1e-3)
        l_reg = self.optimizer.get("lambda",0)

        for epoch in range(epochs):
            mini_batches = self.create_mini_batches(self.X_, self.y_, batch_size, True) 
            epoch_loss = 0
            batch_cnt = 0
            avg_accuarcy = 0
            epoch_nonzero = 0

            with tqdm(total=X.shape[0], ncols=135, disable = not self.verbose) as pbar:
                for batch in mini_batches: 
                    data, target = batch 
                    target_one_hot = np.array([ [1 if t == c else 0 for c in self.classes_] for t in target])
                    output, all_proba = self.predict_proba(data)
                    loss = self.loss_function(output, target_one_hot)
                    loss_deriv = self.loss_function_deriv(output, target_one_hot)
                    
                    epoch_loss += np.mean(loss)
                    accuracy = accuracy_score(target, self.proba_to_class(output))*100.0
                    avg_accuarcy += accuracy
                    batch_cnt += 1
                    
                    if self.weights_per_class:
                        directions = np.mean(all_proba*loss_deriv,axis=1)
                    else:
                        directions = np.mean(all_proba*loss_deriv,axis=(1,2))
                    
                    tmp_w = self.estimator_weights_ - alpha*directions
                    sign = np.sign(tmp_w)
                    tmp_w = np.abs(tmp_w)-l_reg
                    # self.weights = sign*np.maximum(tmp_w,np.zeros_like(tmp_w))
                    self.estimator_weights_ = sign*np.maximum(tmp_w,0)
                    epoch_nonzero += np.count_nonzero(self.estimator_weights_)

                    # for j in range(len(self.weights)):
                    #     #h_j = self.estimators_[j].predict_proba(data)
                    #     h_j = all_proba[j]
                    #     if self.weights_per_class:
                    #         step_direction = np.mean(h_j*loss_deriv,axis=0)
                    #     else:
                    #         step_direction = np.mean(h_j*loss_deriv)

                    #     tmp_w = self.weights[j] - alpha*step_direction
                    #     sign = np.sign(tmp_w)
                    #     tmp_w = np.abs(tmp_w)-l_reg
                    #     self.weights[j] = sign*np.maximum(tmp_w,np.zeros_like(tmp_w))
                    #   # self.weights[j] = self.weights[j] - alpha*step_direction

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

                    test_loss = np.mean(self.loss_function(output, target_one_hot))
                    test_accuracy = accuracy_score(self.y_test, self.proba_to_class(output))*100.0
                    outfile.write("{},{},{},{},{},{}\n".format(epoch,epoch_loss/batch_cnt,avg_accuarcy/batch_cnt,np.count_nonzero(self.estimator_weights_),test_loss,test_accuracy))

                    desc = '[{}/{}] loss {:2.4f} train-acc {:2.4f} nonzero {:2.4f} test-loss {:2.4f} test-acc {:2.4f}'.format(
                        epoch, 
                        epochs-1, 
                        epoch_loss/batch_cnt, 
                        avg_accuarcy/batch_cnt,
                        np.count_nonzero(self.estimator_weights_),
                        test_loss,
                        test_accuracy
                    )
                    pbar.set_description(desc)
            # print(histogram(self.weights.flatten(), bins=20, height=20))
            
        