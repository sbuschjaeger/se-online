import os
import numpy as np
import random
from tqdm import tqdm
import time
import gzip
import pickle
import river
import sys

from functools import partial

from sklearn.utils.multiclass import unique_labels

from scipy.special import softmax

from abc import ABC, abstractmethod

class OnlineLearner(ABC):
    def __init__(self,  
                eval_loss = "cross-entropy",
                seed = None,
                verbose = True, 
                shuffle = True,
                out_path = None):
        
        assert eval_loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"

        self.eval_loss = eval_loss
        self.verbose = verbose
        self.shuffle = shuffle
        self.out_path = out_path

        if seed is None:
            self.seed = 1234
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)

    @abstractmethod
    def next(self, data, target):
        pass
    
    @abstractmethod
    def num_trees(self):
        pass
    
    @abstractmethod
    def num_nodes(self):
        pass

    def num_bytes(self):
        # This call does not include any metrics or statistics computed e.g. during training, but is only the size of this object  itself. 
        size = sys.getsizeof(self.eval_loss) + sys.getsizeof(self.verbose) + sys.getsizeof(self.shuffle) + sys.getsizeof(self.out_path) + sys.getsizeof(self.seed)
        if hasattr(self, "classes_"):
            size += sys.getsizeof(self.classes_)
        
        if hasattr(self, "n_classes_"):
            size += sys.getsizeof(self.n_classes_)
        
        if hasattr(self, "n_outputs_"):
            size += sys.getsizeof(self.n_outputs_)
        
        return size

    @abstractmethod
    def predict_proba(self, X):
        pass

    def compute_loss(self, output, target):
        target_one_hot = np.array( [1.0 if target == i else 0.0 for i in range(self.n_classes_)] )
        if self.eval_loss == "mse":
            loss = (output - target_one_hot) * (output - target_one_hot)
        elif self.eval_loss == "cross-entropy":
            p = softmax(output)
            loss = -target_one_hot*np.log(p + 1e-7)
        elif self.eval_loss == "hinge2":
            zeros = np.zeros_like(target_one_hot)
            loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
        else:
            raise "Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.eval_loss)
        return loss.mean() # For multiclass problems we use the mean over all classes


    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_
        
        river_metrics = {
            "accuracy":river.metrics.Accuracy(),
            "kappa":river.metrics.CohenKappa(),
            # See https://link.springer.com/content/pdf/10.1007/978-3-642-40988-2_30.pdf
            "kappaM":river.metrics.KappaM(),
            "kappaT":river.metrics.KappaT(),
        }

        metrics = {
            "item_cnt" : np.arange(1, len(X) + 1)
        }

        n = 0
        old_size = None
        first_percent = int(X.shape[0] * 0.01)

        with tqdm(total=X.shape[0], ncols=220, disable = not self.verbose) as pbar:
            for i,(x,y) in enumerate(zip(X,y)):
                output = self.predict_proba(x)
                n_nodes = self.num_nodes()
                n_trees = self.num_trees()
                
                # Asking for the size of an objectiv is somewhat expensive. For MOA (=Java) we need to iterate over the entire
                # object hierarchy and for SKLearn we need to (un)pickle everything. To speed things up a bit we compute
                # the size of the model for every percent of data consumed. Since there is probably the most variability in the 
                # size of the objects we also evaluate it for each item in the first percent of the data.
                if (i < first_percent) or i % first_percent == 0:
                    old_size = self.num_bytes()
                
                n_bytes = old_size

                # Update Model                    
                start_time = time.time()
                self.next(x, y)
                item_time = time.time() - start_time
                
                ypred = output.argmax()
                
                item_metrics = {
                    "loss":self.compute_loss(output, y),
                    "time":item_time,
                    "num_nodes":n_nodes,
                    "num_trees":n_trees,
                    "num_bytes":n_bytes
                }
                for key, rm in river_metrics.items():
                    rm.update(y,ypred)
                    item_metrics[key] = rm.get()

                # See https://link.springer.com/content/pdf/10.1007/s10994-014-5441-4.pdf
                item_metrics["kappaC"] = np.sqrt( np.maximum(item_metrics["kappa"],0) * np.maximum(item_metrics["kappaT"], 0) )
                
                # Extract statistics and also compute cumulative sum for plotting later
                for key,val in item_metrics.items():
                    if key not in metrics:
                        metrics[key] = np.zeros( len(X) )

                    metrics[key][n] = val
                    
                    if key + "_sum" not in metrics:
                        metrics[key + "_sum"] = np.zeros( len(X) )
                        metrics[key + "_sum"][n] = val
                    else:
                        metrics[key + "_sum"][n] = metrics[key + "_sum"][n - 1] + val
                
                m_str = ""
                for key,val in metrics.items():
                    if "_sum" in key:
                        if key == "num_bytes_sum":
                            n_bytes = int(val[n] / (n + 1))
                            def human_size(n_bytes, units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
                                """ Returns a human readable string representation of bytes """
                                return str(n_bytes) + units[0] if n_bytes < 1024 else human_size(n_bytes>>10, units[1:])
                            

                            m_str += "{} {} ".format(key.split("_sum")[0], human_size(n_bytes))
                        else:
                            m_str += "{} {:2.4f} ".format(key.split("_sum")[0], val[n] / (n + 1))
                
                pbar.update(1)
                n += 1

                desc = '{}'.format(
                    m_str
                )
                pbar.set_description(desc)

            if self.out_path is not None:
                pickle.dump(metrics, gzip.open(os.path.join(self.out_path, "training.npy.gz"), "wb"))
