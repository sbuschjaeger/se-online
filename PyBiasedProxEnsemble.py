import numpy as np
from joblib import Parallel, delayed
import numbers

from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier

from OnlineLearner import OnlineLearner

def to_prob_simplex(x):
    if x is None or len(x) == 0:
        return x
    sorted_x = np.sort(x)
    x_sum = sorted_x[0]
    l = 1.0 - sorted_x[0]
    for i in range(1,len(sorted_x)):
        x_sum += sorted_x[i]
        tmp = 1.0 / (i + 1.0) * (1.0 - x_sum)
        if (sorted_x[i] + tmp) > 0:
            l = tmp 
    
    return [max(xi + l, 0.0) for xi in x]

class PyBiasedProxEnsemble(OnlineLearner):
    def __init__(self,
                max_depth,
                loss = "cross-entropy",
                step_size = 1e-1,
                ensemble_regularizer = None,
                l_ensemble_reg = 0,  
                tree_regularizer = None,
                l_tree_reg = 0,
                normalize_weights = False,
                init_weight = 0,
                scale_batch = 0,
                var_batch = 1.0,
                n_jobs = 1,
                *args, **kwargs
                ):

        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert ensemble_regularizer is None or ensemble_regularizer in ["none","L0", "L1", "hard-L1"], "Currently only {{none,L0, L1, hard-L1}} the ensemble regularizer is supported"
        assert init_weight in ["average","max"] or isinstance(init_weight, numbers.Number), "init_weight should be {{average, max}} or a number"
        assert not isinstance(init_weight, numbers.Number) or (isinstance(init_weight, numbers.Number) and init_weight > 0), "init_weight should be > 0, otherwise it will we removed immediately after its construction."
        assert l_tree_reg >= 0, "l_tree_reg must be greate or equal to 0"
        assert tree_regularizer is None or tree_regularizer in ["node"], "Currently only {{none, node}} regularizer is supported for tree the regularizer."
        
        if "batch_size" in args and args["batch_size"] <= 1:
            print("WARNING: batch_size should be 2 for PyBiasedProxEnsemble for optimal performance, but was {}. Fixing it for you.".format(args["batch_size"]))
            args["batch_size"] = 2

        if "batch_size" in kwargs and kwargs["batch_size"] <= 1:
            print("WARNING: batch_size should be 2 for PyBiasedProxEnsemble for optimal performance, but was {}. Fixing it for you.".format(kwargs["batch_size"]))
            kwargs["batch_size"] = 2

        if ensemble_regularizer == "hard-L1" and l_ensemble_reg < 1:
            print("WARNING: You set l_ensemble_reg to {}, but regularizer is hard-L1. In this mode, l_ensemble_reg should be an integer 1 <= l_ensemble_reg <= max_trees where max_trees is the number of estimators trained by base_estimator!".format(l_ensemble_reg))

        if (l_ensemble_reg > 0 and (ensemble_regularizer == "none" or ensemble_regularizer is None)):
            print("WARNING: You set l_ensemble_reg to {}, but regularizer is None. Ignoring l_ensemble_reg!".format(l_ensemble_reg))
            l_ensemble_reg = 0
            
        if (l_ensemble_reg == 0 and (ensemble_regularizer != "none" and ensemble_regularizer is not None)):
            print("WARNING: You set l_ensemble_reg to 0, but choose regularizer {}.".format(ensemble_regularizer))

        super().__init__(*args, **kwargs)

        
        self.step_size = step_size
        self.loss = loss
        self.normalize_weights = normalize_weights
        self.init_weight = init_weight
        self.ensemble_regularizer = ensemble_regularizer
        self.l_ensemble_reg = l_ensemble_reg
        self.tree_regularizer = tree_regularizer
        self.l_tree_reg = l_tree_reg
        self.normalize_weights = normalize_weights
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.scale_batch = scale_batch
        self.var_batch = var_batch
        self.estimators_ = []
        self.estimator_weights_ = []
        self.dt_seed = self.seed
    
    def _individual_proba(self, X):
        assert self.estimators_ is not None, "Call fit before calling predict_proba!"
        # def single_predict_proba(h,X):
        #     return h.predict_proba(X)
        
        # TODO MAKE SURE THAT THE ORDER OF H FITS TO ORDER OF WEIGHTS
        # all_proba = Parallel(n_jobs=self.n_jobs, backend="threading")(
        #     delayed(single_predict_proba) (h,X) for h in self.estimators_
        # )
        all_proba = []

        for e in self.estimators_:
            tmp = np.zeros(shape=(X.shape[0], self.n_classes_), dtype=np.float32)
            tmp[:, e.classes_.astype(int)] += e.predict_proba(X)
            all_proba.append(tmp)

        #all_proba = np.array([h.predict_proba(X) for h in self.estimators_])
        return np.array(all_proba)

    def _combine_proba(self, all_proba):
        scaled_prob = np.array([w * p for w,p in zip(all_proba, self.estimator_weights_)])
        combined_proba = np.sum(scaled_prob, axis=0)
        return combined_proba

    def predict_proba(self, X):
        if (len(self.estimators_)) == 0:
            return np.zeros((X.shape[0], self.n_classes_))
        else:
            all_proba = self._individual_proba(X)
            return self._combine_proba(all_proba)

    def next(self, data, target, train = False, new_epoch = False):
        if (len(self.estimators_)) == 0:
            output = np.zeros((data.shape[0], self.n_classes_))
        else:
            all_proba = self._individual_proba(data)
            output = self._combine_proba(all_proba)

        # Compute the appropriate loss. 
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
            loss_deriv = - 2 * target_one_hot * np.maximum(1.0 - target_one_hot * output, zeros) 
        else:
            raise "Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.loss)
        
        # Compute the appropriate ensemble_regularizer
        if self.ensemble_regularizer == "L0":
            loss = np.mean(loss) + self.l_ensemble_reg * np.linalg.norm(self.estimator_weights_,0)
        elif self.ensemble_regularizer == "L1":
            loss = np.mean(loss) + self.l_ensemble_reg * np.linalg.norm(self.estimator_weights_,1)
        else:
            loss = np.mean(loss) 
        
        # Compute the appropriate tree_regularizer
        if self.tree_regularizer == "node":
            loss += self.l_tree_reg * np.sum( [ (w * est.tree_.node_count) for w, est in zip(self.estimator_weights_, self.estimators_)] )
        
        if train:
            if len(self.estimators_) > 0:
                # Compute the gradient for the loss
                directions = np.mean(all_proba*loss_deriv,axis=(1,2))

                 # Compute the gradient for the tree regularizer
                if self.tree_regularizer:
                    node_deriv = self.l_tree_reg * np.array([ est.tree_.node_count for est in self.estimators_])
                else:
                    node_deriv = 0

                # Perform the gradient step. Note that L0 / L1 regularizer is performed via the prox operator 
                # and thus performed _after_ this update.
                tmp_w = self.estimator_weights_ - self.step_size*directions - self.step_size*node_deriv
                
                # Compute the prox step. 
                if self.ensemble_regularizer == "L0":
                    tmp = np.sqrt(2 * self.l_ensemble_reg * self.step_size)
                    tmp_w = np.array([0 if abs(w) < tmp else w for w in tmp_w])
                elif self.ensemble_regularizer == "L1":
                    sign = np.sign(tmp_w)
                    tmp_w = np.abs(tmp_w) - self.step_size*self.l_ensemble_reg
                    tmp_w = sign*np.maximum(tmp_w,0)
                elif self.ensemble_regularizer == "hard-L1":
                    top_K = np.argsort(tmp_w)[-self.l_ensemble_reg:]
                    tmp_w = np.array([w if i in top_K else 0 for i,w in enumerate(tmp_w)])
            else:
                tmp_w = []
            
            # add random gaussian noise to the current batch so increase the number of training data.
            if self.scale_batch > 0 and self.var_batch > 0:
                tmp_data = []
                tmp_label = []
                for i in range(int(self.scale_batch * data.shape[0])):
                    xidx = np.random.choice(range(data.shape[0]))
                    n = np.random.normal(loc = 0.0, scale = self.var_batch, size = 1)
                    tmp_data.append(data[xidx,:] + n)
                    tmp_label.append(target[xidx])

                tmp_data = np.array(tmp_data)
                tmp_label = np.array(tmp_label)
                data = np.vstack([data, tmp_data])
                target = np.hstack([target, tmp_label])


            if (len(set(target)) > 1):
                # Fit a new tree on the current batch. 
                # class_weight = {}
                # for i in range(self.n_classes_):
                #     class_weight[i] = 1.0

                tree = DecisionTreeClassifier(max_depth = self.max_depth, random_state=self.dt_seed, splitter="best", criterion="entropy")
                #, class_weight = class_weight) #, max_features=1)
                self.dt_seed += 1
                tree.fit(data, target)

                if len(self.estimator_weights_) == 0:
                    tmp_w = np.array([1.0])
                else:
                    if self.init_weight == "average":
                        tmp_w = np.append(tmp_w, [sum(tmp_w)/len(tmp_w)])
                    elif self.init_weight == "max":
                        tmp_w = np.append(tmp_w, [max(tmp_w)])
                    else:
                        tmp_w = np.append(tmp_w, [self.init_weight])

                self.estimators_.append(tree)
            else:
                # TODO WHAT TO DO IF ONLY ONE LABEL IS IN THE CURRENT BATCH?
                pass

            # If set, normalize the weights. Note that we use the support of tmp_w for the projection onto the probability simplex
            # as described in http://proceedings.mlr.press/v28/kyrillidis13.pdf
            # Thus, we first need to extract the nonzero weights, project these and then copy them back into corresponding array
            if self.normalize_weights:
                nonzero_idx = np.nonzero(tmp_w)[0]
                nonzero_w = tmp_w[nonzero_idx]
                nonzero_w = to_prob_simplex(nonzero_w)
                self.estimator_weights_ = np.zeros((len(tmp_w)))
                for i,w in zip(nonzero_idx, nonzero_w):
                    self.estimator_weights_[i] = w
            else:
                self.estimator_weights_ = tmp_w
            
            # Remove all trees with zero weight after prox and projection onto the prob. simplex. 
            new_est = []
            new_w = []
            for h, w in zip(self.estimators_, self.estimator_weights_):
                if w > 0:
                    new_est.append(h)
                    new_w.append(w)

            self.estimators_ = new_est
            self.estimator_weights_ = new_w

            n_updates = 1
        else:
            n_updates = 0
            
        return {"loss": loss, "num_trees": self.num_trees(), "num_parameters":self.num_parameters()}, output, n_updates

    def num_trees(self):
        return np.count_nonzero(self.estimator_weights_)

    def num_parameters(self):
        return sum( [ est.tree_.node_count if w != 0 else 0 for w, est in zip(self.estimator_weights_, self.estimators_)] )

    def fit(self, X, y, sample_weight = None):
        super().fit(X, y, sample_weight)
