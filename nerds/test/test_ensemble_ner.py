from nose.tools import assert_equal, assert_true

from nerds.models import DictionaryNER, CrfNER, SpacyNER, EnsembleNER
from nerds.utils import load_data_and_labels

def test_ensemble_ner():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    estimators = [
        (DictionaryNER(), {}),
        (CrfNER(), {"max_iterations": 1}),
        (SpacyNER(), {"num_epochs": 1})
    ]
    model = EnsembleNER()
    model.fit(X, y, estimators=estimators)
    y_pred = model.predict(X)
    assert_equal(len(y), len(y_pred), "Number of predicted and label documents must be same.")
    assert_equal(len(y[0]), len(y_pred[0]), "Number of predicted and label tags must be same.")

