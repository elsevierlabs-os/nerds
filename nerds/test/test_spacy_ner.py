from nose.tools import assert_equal, assert_true

from nerds.models import SpacyNER
from nerds.utils import load_data_and_labels

import shutil

def test_spacy_ner():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    model = SpacyNER()
    model.fit(X, y)
    model.save("nerds/test/data/models")
    model_r = model.load("nerds/test/data/models")
    y_pred = model_r.predict(X)
    assert_equal(y, y_pred, "Label and prediction must be equal")
    assert_equal(1.0, model.score(X, y))
    shutil.rmtree("nerds/test/data/models")
