from nose.tools import assert_equal

from nerds.core.model.evaluate.score import calculate_precision_recall_f1score
from nerds.core.model.input import Annotation, AnnotatedDocument


def test_calculate_precision_recall_f1score():
    content = b"The quick brown fox jumps over the lazy dog."
    annotations = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42))
    ]
    annotated_document = AnnotatedDocument(content, annotations=annotations)

    annotations_pred_perfect = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42))
    ]
    annotated_document_perfect = AnnotatedDocument(
        content, annotations=annotations_pred_perfect)

    annotations_pred_false_pos = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("over", "ANIMAL", (26, 29)),
        Annotation("dog", "ANIMAL", (40, 42)),
    ]
    annotated_document_false_pos = AnnotatedDocument(
        content, annotations=annotations_pred_false_pos)

    annotations_pred_false_neg = [
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42))
    ]
    annotated_document_false_neg = AnnotatedDocument(
        content, annotations=annotations_pred_false_neg)

    assert_equal(calculate_precision_recall_f1score(
        [annotated_document_perfect], [annotated_document]),
        (1.0, 1.0, 1.0)
    )
    assert_equal(calculate_precision_recall_f1score(
        [annotated_document_false_pos], [annotated_document]),
        (0.75, 1.0, 1.5 / 1.75)
    )
    assert_equal(calculate_precision_recall_f1score(
        [annotated_document_false_neg], [annotated_document]),
        (1.0, 2.0 / 3.0, 0.8)
    )
    assert_equal(calculate_precision_recall_f1score(
        [annotated_document_false_pos], [annotated_document], "COLOR"),
        (1.0, 1.0, 1.0)
    )
