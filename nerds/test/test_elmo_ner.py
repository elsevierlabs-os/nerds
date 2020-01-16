from nose.tools import assert_equal, assert_true

from nerds.models import ElmoNER
from nerds.utils import load_data_and_labels

import numpy as np
import shutil

def test_elmo_ner():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    # there are 28 unique words in our "vocabulary"
    embeddings = np.random.random((28, 100))
    model = ElmoNER(embeddings=embeddings, max_iter=1)
    model.fit(X, y)
    model.save("nerds/test/data/models")
    model_r = model.load("nerds/test/data/models")
    y_pred = model_r.predict(X)
    # there is not enough data to train this model properly, so decent
    # asserts are unlikely to succeed.
    assert_equal(len(y), len(y_pred), "Number of labels and predictions must be equal.")
    shutil.rmtree("nerds/test/data/models")