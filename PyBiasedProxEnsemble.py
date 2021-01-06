import numpy as np
from joblib import Parallel, delayed
import copy

from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier

from OnlineLearner import OnlineLearner

class PyBiasedProxEnsemble(OnlineLearner):
    def __init__(self,
                #base_estimator,
                max_depth,
                step_size = 1e-1,
                l_reg = 0,  
                regularizer = "L1",
                loss = "cross-entropy",
                init_weight = 0,
                max_trees = 0,
                n_jobs = 1,
                *args, **kwargs
                ):

        assert loss in ["mse","cross-entropy"], "Currently only {{mse, cross-entropy}} loss is supported"
        assert regularizer is None or regularizer in ["none","L0", "L1", "prob"], "Currently only {{none, L0, L1, prob}} regularizer is supported"
        
        super().__init__(*args, **kwargs)

        if (l_reg > 0 and (regularizer == "none" or regularizer is None)):
            print("WARNING: You set l_reg to {}, but regularizer is None. Ignoring l_reg!".format(l_reg))
            l_reg = 0

        if (l_reg == 0 and (regularizer != "none" and regularizer is not None and regularizer != "prob")):
            print("WARNING: You set l_reg to 0, but choose regularizer {}.".format(regularizer))
        
        self.step_size = step_size
        self.loss = loss
        self.init_weight = init_weight
        self.l_reg = l_reg
        self.regularizer = regularizer
        self.max_depth = max_depth
        self.max_trees = max_trees
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.estimator_weights_ = []
        self._seed = self.seed
    
    def _individual_proba(self, X):
        assert self.estimators_ is not None, "Call fit before calling predict_proba!"

        def single_predict_proba(h,X):
            return h.predict_proba(X)

        all_proba = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(single_predict_proba) (h,X) for h in self.estimators_
        )
        all_proba = np.array(all_proba)
        return all_proba

    def _combine_proba(self, all_proba):
        scaled_prob = np.array([w * p for w,p in zip(all_proba, self.estimator_weights_)])
        combined_proba = np.sum(scaled_prob, axis=0)
        return combined_proba

    def predict_proba(self, X):
        all_proba = self._individual_proba(X)
        return self._combine_proba(all_proba)

    def next(self, data, target, train = False, new_epoch = False):
        if self.max_trees == 0 or len(self.estimators_) < self.max_trees:
            tree = DecisionTreeClassifier(max_depth = self.max_depth, random_state=self._seed) #, max_features=1)
            tree.fit(data, target)

            self.estimator_weights_.append(self.init_weight)
            self.estimators_.append(tree)
            self._seed += 1

        all_proba = self._individual_proba(data)
        output = self._combine_proba(all_proba)

        if self.loss == "mse":
            target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
            loss = (output - target_one_hot) * (output - target_one_hot)
            loss_deriv = 2 * (output - target_one_hot)
        elif self.loss == "cross-entropy":
            target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
            p = softmax(output, axis=1)
            loss = -target_one_hot*np.log(p + 1e-7)
            m = target.shape[0]
            loss_deriv = softmax(output, axis=1)
            loss_deriv[range(m),target_one_hot.argmax(axis=1)] -= 1
        else:
            raise "Currently only the losses {{cross-entropy, mse}} are supported, but you provided: {}".format(self.loss)
        
        if self.regularizer == "L0":
            loss = np.mean(loss) + self.l_reg * np.linalg.norm(self.estimator_weights_,0)
        elif self.regularizer == "L1":
            loss = np.mean(loss) + self.l_reg * np.linalg.norm(self.estimator_weights_,1)
        else:
            loss = np.mean(loss) 

        if train:
            directions = np.mean(all_proba*loss_deriv,axis=(1,2))
            tmp_w = self.estimator_weights_ - self.step_size*directions 
            
            if self.regularizer == "L0":
                tmp = np.sqrt(2 * self.l_reg * self.step_size)
                self.estimator_weights_ = [0 if abs(w) < tmp else w for w in tmp_w]
            elif self.regularizer == "L1":
                sign = np.sign(tmp_w)
                tmp_w = np.abs(tmp_w) - self.step_size*self.l_reg
                self.estimator_weights_ = sign*np.maximum(tmp_w,0)
            elif self.regularizer == "prob":
                sorted_w = sorted(tmp_w, reverse=False)
                w_sum = sorted_w[0]
                l = 1.0 - sorted_w[0]
                for i in range(1,len(sorted_w)):
                    w_sum += sorted_w[i]
                    tmp = 1.0 / (i + 1.0) * (1.0 - w_sum)
                    if (sorted_w[i] + tmp) > 0:
                        l = tmp 
                
                self.estimator_weights_ = [max(w + l, 0.0) for w in tmp_w]
                # print("AFTER PROB:", sum(self.estimator_weights_))
                # print("AFTER PROB:", self.estimator_weights_)
            else:
                self.estimator_weights_ = tmp_w

            new_est = []
            new_w = []
            for h, w in zip(self.estimators_, self.estimator_weights_):
                # TODO Algo hängt in lokalem minimum fest? 
                # - Jeder Baum bekommt min X updates before delete?
                if w > 0:
                    new_est.append(h)
                    new_w.append(w)

            # new_w = [max(w,1.0/len(new_w)) for w in new_w]

            self.estimators_ = new_est
            self.estimator_weights_ = new_w
            # if len(self.estimator_weights_) > 0:
            #     # TODO THIS IS SUPER BRUTAL!
            #     self.estimator_weights_ = list(self.estimator_weights_ / np.sum(self.estimator_weights_))

            n_updates = 1
        else:
            n_updates = 0
            
        return {"loss": loss, "num_trees": self.num_trees(), "num_parameters":self.num_parameters()}, output, n_updates

    def num_trees(self):
        return np.count_nonzero(self.estimator_weights_)

    def num_parameters(self):
        # TODO THIS IS NOT REALLY MEANINGFUL
        return np.count_nonzero(self.estimator_weights_)

    def fit(self, X, y, sample_weight = None):
        super().fit(X, y, sample_weight)
