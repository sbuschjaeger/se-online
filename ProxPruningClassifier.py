import numpy as np
from joblib import Parallel, delayed
import copy

from scipy.special import softmax

from OnlineLearner import OnlineLearner

class ProxPruningClassifier(OnlineLearner):
    def __init__(self,
                base_estimator,
                step_size = 1e-1,
                l_reg = 0,  
                loss = "cross-entropy",
                init_weight = 0,
                n_jobs = 8,
                fast_fit = True,
                *args, **kwargs
                ):

        assert loss in ["mse","cross-entropy", "hinge2"], "Currently only {mse, cross-entropy, hinge2} loss is supported"
        assert base_estimator is not None, "base_estimator must be a valid base model to be fitted"
        
        super().__init__(*args, **kwargs)
        self.base_estimator = copy.deepcopy(base_estimator)
        self.step_size = step_size
        self.loss = loss
        self.init_weight = init_weight
        self.l_reg = l_reg
        self.n_jobs = n_jobs
        self.estimators_ = None
        self.estimator_weights_ = None
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
        scaled_prob = np.array([w*p for w,p in zip(all_proba, self.estimator_weights_)])
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
            target_hinge = np.array( [ [1.0 if y == i else -1.0 for i in range(self.n_classes_)] for y in target] )
            zeros = np.zeros(target_hinge.shape)
            loss = np.maximum(1.0 - target_hinge*output, zeros)**2
            loss_deriv = - 2 * np.maximum(1.0 - target_hinge*output, zeros)
        else:
            raise "Currently only the three losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.loss)
        
        loss = np.mean(loss) + self.l_reg * np.linalg.norm(self.estimator_weights_,1)
        
        if train:
            directions = np.mean(all_proba*loss_deriv,axis=(1,2))
            tmp_w = self.estimator_weights_ - self.step_size*directions
            sign = np.sign(tmp_w)
            tmp_w = np.abs(tmp_w) - self.l_reg
            self.estimator_weights_ = sign*np.maximum(tmp_w,0)

        return {"loss": loss, "num_trees": self.num_trees(), "num_parameters":self.num_parameters()}, output, 1

    def num_trees(self):
        return np.count_nonzero(self.estimator_weights_)

    def num_parameters(self):
        # TODO THIS IS NOT REALLY MEANINGFUL
        return np.count_nonzero(self.estimator_weights_)

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
