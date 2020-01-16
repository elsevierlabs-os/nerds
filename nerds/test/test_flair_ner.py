import warnings
warnings.filterwarnings("ignore")

from nose.tools import assert_equal, assert_true

from nerds.models import FlairNER
from nerds.utils import load_data_and_labels

import shutil

def test_flair_ner():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    model = FlairNER("nerds/test/data/models", max_iter=1)
    model.fit(X, y)
    model.save("nerds/test/data/models")
    model_r = model.load("nerds/test/data/models")
    y_pred = model_r.predict(X)
    # FLAIR NER needs more data to train than provided, so pointless testing
    # for prediction quality, just make sure prediction produces something sane
    assert_equal(len(y), len(y_pred), "Size of Label and prediction must be equal")
    assert_equal(len(y[0]), len(y_pred[0]), "Size of first Label and prediction must be equal")
    shutil.rmtree("nerds/test/data/models")
