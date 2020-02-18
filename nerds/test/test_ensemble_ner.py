import warnings
warnings.filterwarnings("ignore")

from nose.tools import assert_equal, assert_true

from nerds.models import DictionaryNER, CrfNER, SpacyNER, EnsembleNER
from nerds.utils import load_data_and_labels

from sklearn.ensemble import VotingClassifier

def test_ensemble_ner():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    estimators = [
        ("dict_ner", DictionaryNER()),
        ("crf_ner", CrfNER(max_iter=1)),
        ("spacy_ner", SpacyNER(max_iter=1))
    ]
    model = EnsembleNER(estimators=estimators)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert_equal(len(y), len(y_pred), "Number of predicted and label documents must be same.")
    assert_equal(len(y[0]), len(y_pred[0]), "Number of predicted and label tags must be same.")


def test_ensemble_ner_multithreaded():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    estimators = [
        ("dict_ner", DictionaryNER()),
        ("crf_ner", CrfNER(max_iter=1)),
        ("spacy_ner", SpacyNER(max_iter=1))
    ]
    model = EnsembleNER(estimators=estimators, n_jobs=-1)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert_equal(len(y), len(y_pred), "Number of predicted and label documents must be same.")
    assert_equal(len(y[0]), len(y_pred[0]), "Number of predicted and label tags must be same.")
