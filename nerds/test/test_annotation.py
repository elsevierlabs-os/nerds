from nose.tools import assert_equal, assert_true

from nerds.core.model.input.annotation import Annotation


def test_to_inline_string():
    annotation = Annotation("pizza", "FOOD", (1, 6))
    assert_equal(annotation.to_inline_string(), "FOOD[pizza]")


def test_comparison_operators():
    annotation_1 = Annotation("pizza", "FOOD", (1, 6))
    annotation_2 = Annotation("pizza", "FOOD", (1, 6))
    annotation_3 = Annotation("cupcake", "FOOD", (9, 16))
    annotation_4 = Annotation("milk", "FOOD", (7, 11))

    assert_true(annotation_3 > annotation_1)
    assert_true(annotation_4 < annotation_3)

    assert_true(annotation_3 >= annotation_1)
    assert_true(annotation_4 <= annotation_3)

    assert_true(annotation_1 == annotation_2)


def test_hash():
    annotation_1 = Annotation("pizza", "FOOD", (1, 6))
    annotation_2 = Annotation("pizza", "FOOD", (1, 6))

    assert_equal(hash(annotation_1), hash(annotation_2))


def test_str():
    annotation = Annotation("pizza", "FOOD", (1, 6))
    assert_equal(str(annotation), "1,6 FOOD[pizza]")
