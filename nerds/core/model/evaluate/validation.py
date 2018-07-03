import numpy as np
from numpy.random import shuffle
from sklearn.base import clone
from sklearn.model_selection import KFold

from nerds.core.model.evaluate.score import calculate_precision_recall_f1score


class KFoldCV(object):
    """ Wrapper class that offers k-fold cross validation functionality directly
        on `AnnotatedDocument` objects.

        It accepts an `NERModel` object as input along with the cross
        validation parameters, therefore a `KFoldCV` instance needs to be
        created for every different model (instead of passing the model in the
        `cross_validate` function directly as parameter). The reason for that
        is that we may want to hold model-specific metadata for every model
        e.g. for visualization purposes.

        Attributes:
            ner_model (NERModel): An NER model.
            k (int, optional): The number of folds in the k-fold cross
                validation. If 1, then `eval_split` will be used to determine
                the split. Defaults to 10.
            eval_split (float, optional): Only considered when `k = 1`.
                Determines the split percentage of the train-test sets,
                `eval_split * len(X)` is used for training, and the rest for
                test. Defaults to 0.8.
            entity_label (str, optional): The entity label for which the
                precision, recall, and f1-score metrics are calculated.
                Defaults to None, which means all the available entities.
            shuffle_data (bool, optional): Whether to shuffle the data before
                the cross validation. Defaults to True.
    """

    def __init__(self, ner_model, k=10, eval_split=0.8, entity_label=None,
                 shuffle_data=True):
        self.ner_model = ner_model
        self.k = k
        self.eval_split = eval_split
        self.entity_label = entity_label
        self.shuffle_data = shuffle_data

    def cross_validate(self, X, hparams):
        """ Method that performs k-fold cross validation on a set of annotated
            documents.

            Args:
                X (list(AnnotatedDocument)): A list of annotated documents.
                hparams (dict): The hyperparameters of the model, to be passed
                    in the `fit` method.

            Returns:
                float: The average f1-score calculated across `k` experiments.
        """
        X = np.asarray(X)
        if self.shuffle_data:
            shuffle(X)

        if self.k == 1:
            train_test_split = int(len(X) * self.eval_split)
            X_train = X[:train_test_split]
            X_test = X[train_test_split:]
            return self._evaluate_once(X_train, X_test, hparams)

        kfold = KFold(n_splits=self.k)
        f = 0.
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            f += self._evaluate_once(X_train, X_test, hparams)

        return f / self.k

    def _evaluate_once(self, X_train, X_test, hparams):
        """ Helper function to evaluate the NERModel on a set of data. """
        base_estimator = clone(self.ner_model)
        base_estimator.fit(X_train, **hparams)

        X_pred = base_estimator.transform(X_test)
        p, r, f = calculate_precision_recall_f1score(
            X_pred, X_test, entity_label=self.entity_label)
        return f
