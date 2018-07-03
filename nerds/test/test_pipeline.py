from nose.tools import assert_true, assert_equal

from nerds.core.model.input import DataInput


def test_pipeline_integration():
    data_input = DataInput(".", annotated=False)

    assert_true(hasattr(data_input, "fit"))
    assert_true(hasattr(data_input, "transform"))
    assert_true(hasattr(data_input, "fit_transform"))


def test_data_input():
    data_input = DataInput("nerds/test/data/not_annotated", annotated=False)
    doc_1, doc_2 = data_input.transform()
    assert_equal(doc_1.plain_text_, "This is a great car!")
    assert_equal(doc_2.plain_text_, "I love my cats.")
