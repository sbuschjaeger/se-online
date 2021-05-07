import numpy as np
import random
import copy
from numpy.core.fromnumeric import argsort
from scipy.io.arff.arffread import Attribute
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from scipy.special import softmax

from OnlineLearner import OnlineLearner

# Boiler plate stuff to start the module
import jpype
import jpype.imports
from jpype.types import *

def create_moa_model(base_model, model_params):
    def dict_to_string(d):
        s = ""
        for key, value in d.items():
            if key == MoaModel.MOA_EMPTY_PLACEHOLDER:
                dash = ""
                key = ""
            else:
                dash = "-"

            if isinstance(value, dict):
                s += " {}{} ({} )".format(dash, key, dict_to_string(value))
            else:
                s += " {}{} {}".format(dash, key, value)
        return s

    # Make sure to use () so that cliStringToObject correctly parses sub-options
    base_model += dict_to_string(model_params)
    # print(base_model)
    # for key, value in model_params.items():
    #     base_model += " -{} ({})".format(key, value)

    from com.github.javacliparser import ClassOption
    from java.lang import Object

    # So this is a little hacky here. We want to use the CLI interface to (correctly) pass options to 
    # MOA and let MOA take care of creating objects for us etc. 
    # If you look at the following code (https://github.com/Waikato/moa/blob/740e8ca7db30541d6d5deb4550eb3bdb557fa82f/moa/src/main/java/moa/MakeObject.java) we can basically call
    #
    # ClassOption.cliStringToObject(cliString.toString(), Object.class, null);
    #
    # Which seems nice. But since class is a reserved keyword in python this wont work due to syntax errors. Hence
    # we create a tmp object and get is class via getClass() 
    tmp = Object()
    return ClassOption.cliStringToObject(base_model, tmp.getClass(), None)

    # if base_model == "HoeffdingTreeClassifier":
    #     from moa.classifiers.trees import HoeffdingTree 
    #     base_model = HoeffdingTree()
    # elif base_model == "ExtremelyFastDecisionTreeClassifier":
    #     from moa.classifiers.trees import EFDT
    #     base_model = EFDT()
    # elif base_model == "BaggingClassifier":
    #     from moa.classifiers.meta import OzaBag
    #     base_model = OzaBag()

    # for key, value in model_params.items():
    #     getattr(base_model, key).setValueViaCLIString(value)
    #     # if key == "leafpredictionOption":
    #     #     getattr(base_model, key).setChosenLabel(value)
    #     # else:
    #     #     getattr(base_model, key).setValue(value)
    # return base_model

class MoaModel(OnlineLearner):
    MOA_EMPTY_PLACEHOLDER = "MOA_EMPTY_PLACEHOLDER"

    def __init__(self, moa_model = None, moa_params = {}, nominal_attributes = [], moa_jar = None, *args, **kwargs):

        assert moa_jar is not None, "Please provide the path to the moa jar file `moa.jar`"
        assert moa_model is not None, "moa_model was None. This does not work!"
        #self.model = copy.deepcopy(moa_model)

        # TODO Assert the correct model combinations here

        super().__init__(*args, **kwargs)

        self.nominal_attributes = nominal_attributes
        self.header = None

        jpype.startJVM(classpath=[moa_jar])

        from java.lang import System
        from java.io import PrintStream, File
        # TODO make this platform independent. For Windows use NUL instead of /dev/null. What about MAC?
        # https://stackoverflow.com/questions/56760450/redirect-jar-output-when-called-via-jpype-in-python
        System.setErr(PrintStream(File("/dev/null")))

        #self.model = moa_model
        self.model = create_moa_model(moa_model, moa_params)
        self.model.prepareForUse()

    def predict_proba_one(self, x):
        from com.yahoo.labs.samoa.instances import DenseInstance

        if self.header is None:
            output = np.zeros(self.n_classes_)
        else:
            # A more straight-forward solution would be to directly create a new instance with x, e.g.
            #   dense_instance = DenseInstance(1.0, x)  
            # But this does not always work because a DenseInstance must also reserve space for the label, even though this
            # field is not used at all here. For example, StreamingRandomPatches will (correctly) set the label as missingValue
            # for new examples and thus we need to provide space for that attribute - even though it is missing.
            # This code somewhat adopted from the ArffReader
            dense_instance = DenseInstance(len(x) + 1)
            for i, xi in enumerate(x):
                dense_instance.setValue(i, xi)

            dense_instance.setDataset(self.header)
            #dense_instance.setClassValue(5)
            output = self.model.getPredictionForInstance(dense_instance).getVotes()

            # Okay MOA seems to be a bit weird here and I am super unsure about this now, but from my experiments and the code I read so far the following invariant seems to hold:
            # The class mapping is always [0,1,2,3,...,n_classes-1]. A classifier will always output a prediction vector which is as long as the largest class it has witnessed by now -- regardless of the instance header we constructed. If a classifier has not yet seen any classes it will output an empty string. 
            # https://github.com/Waikato/moa/blob/740e8ca7db30541d6d5deb4550eb3bdb557fa82f/moa/src/main/java/moa/classifiers/meta/OzaBag.java
            output = np.append(np.array(output), np.zeros(self.n_classes_ - len(output)))
            
            # Sometimes MOA produces nan / inf values (especially when using NaiveBayes in leaf nodes of trees). There is a todo about this since the beginning of time (?)in https://github.com/Waikato/moa/blob/740e8ca7db30541d6d5deb4550eb3bdb557fa82f/moa/src/main/java/moa/classifiers/bayes/NaiveBayes.java#L51 but I guess it never occured as a problem in MOA because they use (see e.g. https://github.com/Waikato/moa/blob/740e8ca7db30541d6d5deb4550eb3bdb557fa82f/moa/src/main/java/moa/classifiers/AbstractClassifier.java#L54)
            # 
            # return Utils.maxIndex(getVotesForInstance(inst)) == (int) inst.classValue();
            # 
            # to measure the accuracy and do not compute losses?
            # I try to approximate this by replacing nan/inf values as good as possible

            # TODO this silently assumes that inf should be mapped to 1 which makes sense given that we normalize the output anyway afterwards
            # However, if we have an output such as [inf, 5, 3.1] then inf should probably be mapped to a higher value
            output = np.nan_to_num(output, nan=0, posinf=1, neginf=0)

        # If for some reason the output only contains 0 entries (e.g. no header was yet set because next() was not called yet) then just return the default predictions
        if np.all(output==0):
            return 1.0 / self.n_classes_ * np.ones(self.n_classes_)
        else:
            if sum(output) > 0:
                return output / sum(output)
            else:
                return output

    def predict_proba(self, X):
        if len(X.shape) < 2:
            return self.predict_proba_one(X)
        else:
            proba = []
            for x in X:
                proba.append(self.predict_proba_one(x))
            return np.array(proba)

    def num_trees(self):
        if hasattr(self.model, "ensembleSizeOption"):
            return self.model.ensembleSizeOption.getValue()
        else:
            return 1
        # if hasattr(self.model, "n_models"):
        #     return self.model.n_models
        # else:
        #     return 1

    def num_parameters(self):
        def get_tree_params(tree):
            # Okay for some reason getNodeCount() was not implemented in EFDT and thus we do some reflection stuff to access
            # the protected fields to get the node count. You know, just your typical Java stuff - Yay \o/
            node_cnt = 0
            clazz = tree.getClass()
            for f in ["decisionNodeCount", "activeLeafNodeCount", "inactiveLeafNodeCount"]:
                field = clazz.getDeclaredField(f)
                field.setAccessible(1)
                node_cnt += field.get(tree)
            return node_cnt

        if hasattr(self.model, "ensembleSizeOption"):
            # ensemble is a protected field in meta classifiers so we use the same reflection "trick" from above to access it.
            clazz = self.model.getClass()
            field = clazz.getDeclaredField("ensemble")
            field.setAccessible(1)
            estimators = field.get(self.model)
            node_cnt = 0
            for e in estimators:
                if hasattr(e, "classifier"):
                    node_cnt += get_tree_params(e.classifier)
                else:
                    node_cnt += get_tree_params(e)
            return node_cnt 
        else:
            return get_tree_params(self.model)

    def next(self, data, target):
        from com.yahoo.labs.samoa.instances import DenseInstance
        from com.yahoo.labs.samoa.instances import InstancesHeader
        from com.yahoo.labs.samoa.instances import Instances
        from com.yahoo.labs.samoa.instances import Attribute

        from java.util import ArrayList

        if self.header is None:
            attributes = ArrayList()

            # We compute a one hot encoding of all nominal values and thus they are either 0 or 1
            nominal_values = ArrayList()
            nominal_values.add(0)
            nominal_values.add(1)
            for i in range(len(data)):
                att_name = "att_"+str(i)
                if att_name in self.nominal_attributes:
                    attributes.add(Attribute(att_name, nominal_values))
                else:
                    attributes.add(Attribute(att_name))

            # Make sure that MOA knows the number of classes and their corresponding mapping. 
            label_values = ArrayList()
            for c in self.classes_:
                label_values.add(c)
            attributes.add(Attribute("@label", label_values))
            
            instances = Instances("dummy_stream", attributes, 1)
            instances.setClassIndex(len(data))
            self.header = InstancesHeader(instances)

        # output = self.predict_proba_one(data)

        # The DenseInstance is implemented via a double array which requires space for all features and the label. 
        # Thus increase the data array by one element
        data_plus_one = np.append(data, 0)
        dense_instance = DenseInstance(1.0, data_plus_one)
        dense_instance.setDataset(self.header)
        dense_instance.setClassValue(target)
        self.model.trainOnInstance(dense_instance)

        # output = np.array(output)
        # accuracy = (output.argmax() == target) * 100.0

        # return {"accuracy": accuracy, "num_trees": self.num_trees(), "num_parameters" : self.num_parameters()}, output
