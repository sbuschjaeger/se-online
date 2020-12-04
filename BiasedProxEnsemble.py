import numpy as np
import random
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

# from plotille import histogram
from PyBPE import RandomBiasedProxEnsemble, TrainedBiasedProxEnsemble

class BiasedProxEnsemble:
    def __init__(self,  
                max_depth,
                alpha = 1e-3,
                l_reg = 1e-5,
                loss = "cross-entropy",
                mode = "random",
                init_weight = 0,
                epochs = 50,
                batch_size = 128,
                seed = 1234,
                verbose = True, 
                x_test = None, 
                y_test = None, 
                eval_every = 5):
                        
        assert loss == "cross-entropy", "Currently only cross entropy loss is supported"
        assert mode in ["random", "trained"], "Currently only {random, trained} mode supported"
        assert max_depth >= 1, "max_depth should be at-least 1!"


        self.max_depth = max_depth
        self.alpha = alpha
        self.l_reg = l_reg
        self.loss = loss
        self.mode = mode
        self.init_weight = init_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.x_test = x_test
        self.y_test = y_test
        self.eval_every = eval_every
        self.model = None

        if seed is None:
            self.seed = 1234
        else:
            self.seed= seed

        np.random.seed(seed)
        random.seed(seed)
    
    def predict_proba(self, X):
        assert self.model is not None, "Call fit before calling predict_proba!"
        return np.array(self.model.predict_proba(X))

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

        self.model = RandomBiasedProxEnsemble(self.max_depth, self.n_classes_, self.seed, self.alpha, self.l_reg, self.init_weight, self.mode, self.loss)

        epochs = self.epochs
        batch_size = self.batch_size

        for epoch in range(epochs):
            mini_batches = self.create_mini_batches(self.X_, self.y_, batch_size, True) 
            epoch_loss = 0
            batch_cnt = 0
            avg_accuarcy = 0
            epoch_nonzero = 0

            with tqdm(total=X.shape[0], ncols=135, disable = not self.verbose) as pbar:
                for batch in mini_batches: 
                    data, target = batch 

                    lsum = self.model.next(data, target)
                    output = self.predict_proba(data)
                    
                    epoch_loss += lsum / data.shape[0]
                    epoch_nonzero += self.model.num_trees()
                    accuracy = accuracy_score(target, output.argmax(axis=1))*100.0
                    #accuracy = 0
                    avg_accuarcy += accuracy
                    batch_cnt += 1

                    pbar.update(data.shape[0])
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f} nonzero {:2.4f}'.format(
                        epoch, 
                        epochs-1, 
                        epoch_loss/batch_cnt, 
                        avg_accuarcy/batch_cnt,
                        epoch_nonzero/batch_cnt
                    )
                    pbar.set_description(desc)
                
                if self.x_test is not None and self.y_test is not None:
                    test_accuracy = accuracy_score(self.y_test, self.predict_proba(self.x_test))*100.0
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f} nonzero {:2.4f} test-acc {:2.4f}'.format(
                        epoch, 
                        epochs-1, 
                        epoch_loss/batch_cnt, 
                        avg_accuarcy/batch_cnt,
                        epoch_nonzero/batch_cnt,
                        test_accuracy
                    )
                    pbar.set_description(desc)
