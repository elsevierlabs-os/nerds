from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from nerds.utils import flatten_list

class NERModel(BaseEstimator, ClassifierMixin):
    """ Provides a basic interface to train NER models and annotate documents.

        This is the core class responsible for training models that perform
        named entity recognition, and retrieving named entities from documents.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y):
        """ Train the model using data (X) and labels (y). Return trained model.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Makes predictions using trained model on data (X) and returns them.
        """
        raise NotImplementedError()

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path.
            Should be overridden.
        """
        raise NotImplementedError()

    def load(self, file_path):
        """ Loads a model saved locally. Should be overridden. """
        raise NotImplementedError()

    def score(self, X, y, sample_weights=None):
        """ Returns score for the model based on predicting on (X, y).  This 
            method is needed for GridSearch like operations.

            Parameters
            ----------
            X : list(list(str))
                list of list of tokens.
            y : list(list(str))
                list of list of tags
            sample_weights : list(float), not used

            Returns
            -------
            score: float
                numeric score for estimator.
        """
        y_pred = self.predict(X)
        return accuracy_score(flatten_list(y), flatten_list(y_pred))

