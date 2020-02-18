from nose.tools import assert_equal

from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import Document, AnnotatedDocument
from nerds.core.model.ner.crf import CRF


def test_crf():
    content = b"The quick brown fox jumps over the lazy dog."
    annotations = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42))]
    annotated_document = AnnotatedDocument(content, annotations=annotations)

    # Train.
    mod = CRF()
    mod.fit([annotated_document])

    # Predict. Works!
    content = b"The quick brown fox."
    document = Document(content)

    ann = mod.transform([document])

    # 2 annotations, brown and fox
    assert_equal(len(ann[0].annotations), 2)
    assert_equal(ann[0].annotations[0].text, "brown")
    assert_equal(ann[0].annotations[0].label, "COLOR")
    assert_equal(ann[0].annotations[0].offset, (10, 14))
    assert_equal(ann[0].annotations[1].text, "fox")
    assert_equal(ann[0].annotations[1].label, "ANIMAL")
    assert_equal(ann[0].annotations[1].offset, (16, 18))


def test_crf_multi_term():
    content = b"The dark brown fox jumps over the lazy dog magnificently."
    annotations = [
        Annotation("dark brown", "COLOR", (4, 13)),
        Annotation("fox", "ANIMAL", (15, 17)),
        Annotation("dog", "ANIMAL", (40, 42))]
    annotated_document = AnnotatedDocument(content, annotations=annotations)

    # Train.
    mod = CRF()
    mod.fit([annotated_document])

    # Predict. Works!
    content = b"The dark brown fox."
    document = Document(content)

    ann = mod.transform([document])

    # 2 annotations, dark brown and fox
    assert_equal(len(ann[0].annotations), 2)
    assert_equal(ann[0].annotations[0].text, "dark brown")
    assert_equal(ann[0].annotations[0].offset, (4, 13))
    assert_equal(ann[0].annotations[0].label, "COLOR")
    assert_equal(ann[0].annotations[1].text, "fox")
    assert_equal(ann[0].annotations[1].offset, (15, 17))
    assert_equal(ann[0].annotations[1].label, "ANIMAL")
