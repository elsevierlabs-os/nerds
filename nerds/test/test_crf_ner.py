from nose.tools import assert_equal, assert_true

from nerds.models import CrfNER
from nerds.utils import load_data_and_labels

import shutil

def test_crf_ner():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    model = CrfNER()
    model.fit(X, y)
    model.save("nerds/test/data/models")
    r_model = model.load("nerds/test/data/models")
    y_pred = r_model.predict(X)
    assert_equal(y, y_pred, "Label and prediction must be equal")
    assert_equal(1.0, model.score(X, y))
    shutil.rmtree("nerds/test/data/models")
