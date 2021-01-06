import numpy as np
from joblib import Parallel, delayed
import copy

from scipy.special import softmax

from OnlineLearner import OnlineLearner

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

def n_nodes(model):
    node_cnt = 0
    for est in model.estimators_:
        node_cnt += est.tree_.node_count
    return node_cnt

class ProxPruningClassifier(OnlineLearner):
    def __init__(self,
                base_estimator,
                step_size = 1e-1,
                l_reg = 0,  
                regularizer = "L1",
                node_penalty = 0,
                loss = "cross-entropy",
                normalize_weights = True,
                init_weight = 0,
                n_jobs = 1,
                fast_fit = True,
                *args, **kwargs
                ):

        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert regularizer is None or regularizer in ["none","L0", "L1"], "Currently only {{none,L0, L1}} regularizer is supported"
        assert base_estimator is not None, "base_estimator must be a valid base model to be fitted"
        assert isinstance(base_estimator, (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)), "Only the following base_estimators are currently supported {{RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier}}"
        assert node_penalty >= 0, "node_penalty must be greate or equal to 0"

        super().__init__(*args, **kwargs)

        if (l_reg > 0 and (regularizer == "none" or regularizer is None)):
            print("WARNING: You set l_reg to {}, but regularizer is None. Ignoring l_reg!".format(l_reg))
            l_reg = 0
            
        if (l_reg == 0 and (regularizer != "none" and regularizer is not None)):
            print("WARNING: You set l_reg to 0, but choose regularizer {}.".format(regularizer))

        self.base_estimator = copy.deepcopy(base_estimator)
        self.step_size = step_size
        self.loss = loss
        self.normalize_weights = normalize_weights
        self.init_weight = init_weight
        self.l_reg = l_reg
        self.regularizer = regularizer
        self.n_jobs = n_jobs
        self.estimators_ = None
        self.estimator_weights_ = None
        self.node_penalty = node_penalty
        self.fast_fit = fast_fit
    
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
        if self.normalize_weights:
            sum_w = np.sum(self.estimator_weights_) + 1e-7
            scaled_prob = np.array([w / sum_w * p for p,w in zip(all_proba, self.estimator_weights_)])
        else:
            scaled_prob = np.array([w * p for p,w in zip(all_proba, self.estimator_weights_)])

        combined_proba = np.sum(scaled_prob, axis=0)
        return combined_proba

    def predict_proba(self, X):
        all_proba = self._individual_proba(X)
        return self._combine_proba(all_proba)

    def next(self, data, target, train = False, new_epoch = False):
        if self.fast_fit and train:
            train_idx = [i[0] for i in data]
            all_proba = self.train_preds[:,train_idx,:]
            output = self._combine_proba(all_proba)
        else:
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
        elif self.loss == "hinge2":
            target_one_hot = np.array( [ [1.0 if y == i else -1.0 for i in range(self.n_classes_)] for y in target] )
            zeros = np.zeros_like(target_one_hot)
            loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
            loss_deriv = - 2 * np.maximum(1.0 - target_one_hot * output, zeros)
        else:
            raise "Currently only the losses {{cross-entropy, mse}} are supported, but you provided: {}".format(self.loss)
        
        if self.regularizer == "L0":
            loss = np.mean(loss) + self.l_reg * np.linalg.norm(self.estimator_weights_,0)
        elif self.regularizer == "L1":
            loss = np.mean(loss) + self.l_reg * np.linalg.norm(self.estimator_weights_,1)
        else:
            loss = np.mean(loss) 
        
        if self.node_penalty > 0:
            # w_nodes = np.array([ (w * est.tree_.node_count) for w, est in zip(self.estimator_weights_, self.estimators_)])
            loss += self.node_penalty * np.sum( [ (w * est.tree_.node_count) for w, est in zip(self.estimator_weights_, self.estimators_)] )
            node_deriv = self.node_penalty * np.array([ est.tree_.node_count for est in self.estimators_])
        else:
            node_deriv = 0

        if train:
            directions = np.mean(all_proba*loss_deriv,axis=(1,2))
            if self.normalize_weights:
                sum_w = sum(self.estimator_weights_) + 1e-7
                sums = np.array([(sum_w - w) / sum_w**2 for w in self.estimator_weights_])
                tmp_w = self.estimator_weights_ - self.step_size*directions*sums - self.step_size*node_deriv
            else:
                tmp_w = self.estimator_weights_ - self.step_size*directions - self.step_size*node_deriv
            # if self.normalize_weights:
            #     tmp_w = self.estimator_weights_ - self.step_size*directions - self.step_size*2*0.7*(sum(self.estimator_weights_) - 1.0)
            # else:
            #     tmp_w = self.estimator_weights_ - self.step_size*directions - self.step_size*node_deriv
            
            if self.regularizer == "L0":
                tmp = np.sqrt(2 * self.l_reg * self.step_size)
                self.estimator_weights_ = [0 if abs(w) < tmp else w for w in tmp_w]
            elif self.regularizer == "L1":
                sign = np.sign(tmp_w)
                tmp_w = np.abs(tmp_w) - self.step_size*self.l_reg
                self.estimator_weights_ = sign*np.maximum(tmp_w,0)
            else:
                self.estimator_weights_ = tmp_w

            if self.normalize_weights:
                self.estimator_weights_ = np.array([max(x,0) for x in self.estimator_weights_])

            #print("SUM: ", sum(self.estimator_weights_))
            # if self.normalize_weights:
            #     sorted_w = sorted(tmp_w, reverse=False)
            #     w_sum = sorted_w[0]
            #     l = 1.0 - sorted_w[0]
            #     for i in range(1,len(sorted_w)):
            #         w_sum += sorted_w[i]
            #         tmp = 1.0 / (i + 1.0) * (1.0 - w_sum)
            #         if (sorted_w[i] + tmp) > 0:
            #             l = tmp 
                
            #     self.estimator_weights_ = [max(w + l, 0.0) for w in tmp_w]
                
            n_updates = 1
        else:
            n_updates = 0
            
        return {"loss": loss, "num_trees": self.num_trees(), "num_parameters":self.num_parameters()}, output, n_updates

    def num_trees(self):
        return np.count_nonzero(self.estimator_weights_)

    def num_parameters(self):
        return sum( [ est.tree_.node_count if w != 0 else 0 for w, est in zip(self.estimator_weights_, self.estimators_)] )

    def fit(self, X, y, sample_weight = None):
        model = self.base_estimator.fit(X, y, sample_weight)

        self.estimators_ = model.estimators_
        self.estimator_weights_ = np.ones((len(self.estimators_), )) * self.init_weight
        
        if self.fast_fit:
            Xtmp = np.array([[i] for i in range(X.shape[0])])
            self.train_preds = self._individual_proba(X)
            super().fit(Xtmp, y, sample_weight)
        else:
            super().fit(X, y, sample_weight)
