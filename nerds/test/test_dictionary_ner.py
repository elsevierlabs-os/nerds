from nose.tools import assert_equal, assert_true

from nerds.models import DictionaryNER
from nerds.utils import load_data_and_labels

import shutil

def test_dictionary_ner_from_conll():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    model = DictionaryNER()
    model.fit(X, y)
    model.save("nerds/test/data/models")
    r_model = model.load("nerds/test/data/models")
    y_pred = r_model.predict(X)
    assert_equal(y, y_pred, "Label and prediction must be equal")
    assert_equal(1.0, model.score(X, y))
    shutil.rmtree("nerds/test/data/models")


def test_dictionary_ner_from_dict():
    # load and fit model from dictionary
    xs, ys = [], []
    fdict = open("nerds/test/data/example.ents", "r")
    for line in fdict:
        x, y = line.strip().split('\t')
        xs.append(x)
        ys.append(y)
    fdict.close()
    model = DictionaryNER()
    model.fit(xs, ys, combine_tokens=False)
    # predict using example
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    y_pred = model.predict(X)
    assert_equal(y, y_pred, "Label and prediction must be equal")
    assert_equal(1.0, model.score(X, y))
