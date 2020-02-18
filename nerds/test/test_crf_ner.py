import warnings
warnings.filterwarnings("ignore")

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


def test_crf_ner_with_nondefault_features():
    def my_test_featurizer(sentence):
        return [{"word":token} for token in sentence]

    X, y = load_data_and_labels("nerds/test/data/example.iob")
    model = CrfNER(featurizer=my_test_featurizer)
    model.fit(X, y)
    y_pred = model.predict(X)
    # our features are not good enough to do good predictions, so just
    # check the lengths of labels vs predictions to make sure it worked
    assert_equal(len(y), len(y_pred), "Number of label and predictions must be equal.")
    assert_equal(len(y[0]), len(y_pred[0]), "Size of label and predictions must match (1).")
    assert_equal(len(y[1]), len(y_pred[1]), "Size of label and predictions must match (2).")
