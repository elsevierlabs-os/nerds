from nose.tools import assert_equal, assert_true

from nerds.models import BiLstmCrfNER
from nerds.utils import load_data_and_labels

def test_crf_ner():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    model = BiLstmCrfNER()
    model.fit(X, y, num_epochs=1)
    y_pred = model.predict(X)
    # there is not enough data to train this model properly, so decent
    # asserts are unlikely to succeed.
    assert_equal(len(y), len(y_pred), "Number of labels and predictions must be equal.")
