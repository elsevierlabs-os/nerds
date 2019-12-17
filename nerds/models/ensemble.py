from nerds.models import NERModel
from nerds.utils import get_logger

from sklearn.preprocessing import LabelEncoder

import joblib
import numpy as np
import os

log = get_logger()

class EnsembleNER(NERModel):

    def __init__(self,
            estimators=[],
            weights=None,
            n_jobs=1,
            is_pretrained=False):
        """ Constructor for Voting Ensemble NER.

            Args:
                estimators (list(NERModel, dict(str,obj)), default empty): list 
                    of (NERModels, fit_param) pairs to use in the ensemble. The
                    fit_param is a flat dictionary of named arguments used in
                    fit() for the particular NERModel.
                weights (list(int), default None): sequence of weights to 
                    apply to predicted class labels from each estimator. If
                    None, then predictions from all estimators are treated 
                    equally.
                n_jobs (int, default=1): number of jobs to run in parallel, 
                    default is to single-thread. -1 means to use all available
                    resources.
                is_pretrained (bool, default False): if True, estimators are
                    assumed to be pretrained and fit() is skipped.
        """
        super().__init__()
        # these are set by fit and load, required by predict and save
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs
        self.is_pretrained=is_pretrained


    def fit(self, X, y):
        """ Train ensemble by training underlying NERModels.

            Args:
                X (list(list(str))): list of list of tokens.
                y (list(list(str))): list of list of BIO tags.
        """
        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError("Non-empty list of estimators required to fit ensemble.")
        if self.weights is None:
            self.weights = [1] * len(self.estimators)
        else:
            if len(self.estimators) != len(self.weights):
                raise ValueError("Number of weights must correspond to number of estimators.")

        if self.is_pretrained:
            return self

        fitted_estimators = joblib.Parallel(n_jobs=self.n_jobs, backend="threading")(
            joblib.delayed(self._fit_estimator)(clf, X, y) 
            for name, clf in self.estimators)
        self.estimators = [(name, fitted) for (name, clf), fitted 
            in zip(self.estimators, fitted_estimators)]

        return self


    def predict(self, X):
        """
            Predicts using each estimator in the ensemble, then merges the
            predictions using a voting scheme given by the vote() method
            (subclasses can override voting policy by overriding vote()).

            Args:
                X (list(list(str))): list of list of tokens to predict from.

            Returns:
                ypred (list(list(str))): list of list of BIO tags.
        """
        if self.estimators is None or self.weights is None:
            raise ValueError("Model not ready to predict. Call fit() first, or if using pre-trained models, call fit() with is_pretrained=True")
        
        predictions = joblib.Parallel(n_jobs=self.n_jobs, backend="threading")(
            joblib.delayed(self._predict_estimator)(clf, X) 
            for name, clf in self.estimators)

        return self._vote(predictions)


    def load(model_dirpath):
        raise NotImplementedError()


    def save(model_dirpath):
        raise NotImplementedError()


    def _fit_estimator(self, estimator, X, y):
        fitted_estimator = estimator.fit(X, y)
        return fitted_estimator


    def _predict_estimator(self, estimator, X):
        return estimator.predict(X)


    def _vote(self, predictions):
        """
            Voting mechanism (can be overriden by subclass if desired).

            Args:
                predictions (list(list(list(str)))): a list of list of list of BIO
                    tags predicted by each NER in the ensemble. Each NER outputs
                    a list of list of BIO tags where the outer list corresponds
                    to sentences and the inner list corresponds to tokens.

            Returns:
                voted_predictions (list(list(str))): a list of list of BIO tags.
                Each BIO tag represents the most frequent tag 
        """
        tag2int, int2tag = self._build_label_vocab(predictions)

        best_preds = []
        for row_id in range(len(predictions[0])):

            row_preds = []
            # gather all predictions for this row
            for est_id in range(len(predictions)):
                sent_pred = np.array([tag2int[y] for y in predictions[est_id][row_id]])
                # weighted by weights if any
                for weight in range(self.weights[est_id]):
                    row_preds.append(sent_pred)

            # convert to numpy matrix for performance
            R = np.array(row_preds)

            # we now find the most frequent tag at each position
            B = np.zeros((R.shape[1]), dtype="int32")
            for col_id in range(R.shape[1]):
                col = R[:, col_id]
                values, indices = np.unique(col, return_inverse=True)
                B[col_id] = values[np.argmax(np.bincount(indices))]

            # append the labels associated with the most frequent tags
            best_preds.append([int2tag[x] for x in B.tolist()])

        return best_preds


    def _build_label_vocab(self, predictions):
        """ build lookup table from token to int and back (for performance) """
        tags, tag2int, int2tag = [], {}, {}
        label_encoder = LabelEncoder()
        for est_pred in predictions:
            for sent_pred in est_pred:
                for tok_pred in sent_pred:
                    tags.append(tok_pred)
        label_encoder.fit(tags)
        tag2int = {t:i for i, t in enumerate(label_encoder.classes_)}
        int2tag = {i:t for t, i in tag2int.items()}
        return tag2int, int2tag
