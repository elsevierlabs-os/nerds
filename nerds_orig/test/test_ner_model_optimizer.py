from nose.tools import assert_greater_equal

from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import AnnotatedDocument
from nerds.core.model.ner.crf import CRF
from nerds.core.model.optimize.optimizer import Optimizer
from nerds.core.model.optimize.params import ExactListParam


def test_optimizer():
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
    hparams = {
        "c1": ExactListParam([0.1, 0.9]),
        "c2": ExactListParam([0.1, 0.9])
    }
    optimizer = Optimizer(crf, hparams, "COLOR", cv=3)
    best_estimator, f1score = optimizer.optimize_and_return_best([
        annotated_document_1, annotated_document_2, annotated_document_3
    ])

    assert_greater_equal(f1score, 0.5)
