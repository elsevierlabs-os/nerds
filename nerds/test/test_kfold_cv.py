from nose.tools import assert_equal

from nerds.core.model.evaluate.validation import KFoldCV
from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import AnnotatedDocument
from nerds.core.model.ner.crf import CRF


def test_kfold_cv():
    content = b"The quick brown fox jumps over the lazy dog."
    annotations = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42))]
    annotated_document_1 = AnnotatedDocument(content, annotations=annotations)

    content = b"A brown fox jumps quickly."
    annotations = [
        Annotation("brown", "COLOR", (2, 6)),
        Annotation("fox", "ANIMAL", (8, 10))]
    annotated_document_2 = AnnotatedDocument(content, annotations=annotations)

    content = b"The fox that was brown jumps over a dog that was lazy."
    annotations = [
        Annotation("fox", "ANIMAL", (4, 6)),
        Annotation("brown", "COLOR", (17, 21)),
        Annotation("dog", "ANIMAL", (36, 38))]
    annotated_document_3 = AnnotatedDocument(content, annotations=annotations)

    # Test it with CRF because it's the least time-consuming one to train.
    crf = CRF()
    kfold = KFoldCV(crf, k=3)
    average_f1 = kfold.cross_validate([
        annotated_document_1, annotated_document_2, annotated_document_3
    ], {"max_iterations": 100})

    # The examples and k are selected in a way where this always happens.
    assert_equal(average_f1, 0.5)
