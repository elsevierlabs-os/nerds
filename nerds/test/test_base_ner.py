import warnings
warnings.filterwarnings("ignore")

from nose.tools import assert_equal, assert_raises

from nerds.models import NERModel

def test_fit():
    model = NERModel()
    assert_raises(NotImplementedError, model.fit, [], [])


def test_predict():
    model = NERModel()
    assert_raises(NotImplementedError, model.predict, [])


def test_load():
    model = NERModel()
    assert_raises(NotImplementedError, model.load, "")


def test_save():
    model = NERModel()
    assert_raises(NotImplementedError, model.save, "")


def test_score():
    model = NERModel()
    assert_raises(NotImplementedError, model.score, [], [])
