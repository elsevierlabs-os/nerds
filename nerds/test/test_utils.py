from nose.tools import assert_equal, assert_true

from nerds.utils import *


def test_load_data_and_labels():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    assert_true(len(X) == 2, "There should be 2 sentences in X")
    assert_equal(len(X), len(y), "There should be tags for 2 sentences in y")
    assert_equal(len(X[0]), len(y[0]), "Number of tokens should be equal to number of tags")


def test_flatten_lol():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    yflat = flatten_lol(y, strip_prefix=True)
    assert_equal(36, len(yflat), "There should be 36 tags in all")
    assert_equal(5, len([y for y in yflat if y == "PER"]), "There should be 5 PER tags")
