from nose.tools import assert_equal, assert_raises

from nerds.core.model.input.document import Document
from nerds.core.model.ner.base import NERModel


def test_fit():
    model = NERModel()
    assert_equal(model.fit([]), model)


def test_transform():
    model = NERModel()
    content = b"The quick brown fox jumps over the lazy dog."
    document = Document(content)
    ann_docs = model.transform([document])
    assert_equal(len(ann_docs), 1)
    assert_equal(len(ann_docs[0].annotations), 0)


def test_extract():
    model = NERModel()
    content = b"The quick brown fox jumps over the lazy dog."
    document = Document(content)
    entities = model.extract([document])
    assert_equal(len(entities), 1)
    assert_equal(len(entities[0]), 0)


def test_save():
    model = NERModel()
    assert_raises(NotImplementedError, model.save, "")


def test_load():
    model = NERModel()
    assert_raises(NotImplementedError, model.load, "")
