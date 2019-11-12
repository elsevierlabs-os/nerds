from nose.tools import assert_equal

from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import Document, AnnotatedDocument
from nerds.core.model.ner.spacy import SpaCyStatisticalNER


def test_transform_to_spacy_format():
    content = b"The quick brown fox jumps over the lazy dog."
    annotations = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42))
    ]
    annotated_document = AnnotatedDocument(content, annotations=annotations)

    expected = [(
        "The quick brown fox jumps over the lazy dog.",
        {
            "entities": [
                (10, 15, "COLOR"),
                (16, 19, "ANIMAL"),
                (40, 43, "ANIMAL")
            ]
        }
    )]

    model = SpaCyStatisticalNER()
    transformed = model._transform_to_spacy_format([annotated_document])

    assert_equal(transformed, expected)


def test_spacy():
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

    data = [annotated_document_1, annotated_document_2, annotated_document_3]

    # Train.
    model = SpaCyStatisticalNER()
    model.fit(data, num_epochs=5)

    # Predict. Works!
    content = b"The quick brown fox."
    document = Document(content)

    ann = model.transform([document])

    # 2 annotations, brown and fox
    assert_equal(len(ann[0].annotations), 2)
    assert_equal(ann[0].annotations[0].text, "brown")
    assert_equal(ann[0].annotations[0].label, "COLOR")
    assert_equal(ann[0].annotations[0].offset, (10, 14))
    assert_equal(ann[0].annotations[1].text, "fox")
    assert_equal(ann[0].annotations[1].label, "ANIMAL")
    assert_equal(ann[0].annotations[1].offset, (16, 18))

    entities = model.extract([document])
    assert_equal(len(entities[0]), 2)


def test_spacy_multi_term():
    content = b"The quick dark brown fox jumps over the lazy dog."
    annotations = [
        Annotation("dark brown", "COLOR", (10, 19)),
        Annotation("fox", "ANIMAL", (21, 23)),
        Annotation("dog", "ANIMAL", (45, 47))]
    annotated_document_1 = AnnotatedDocument(content, annotations=annotations)

    content = b"A dark brown fox jumps quickly."
    annotations = [
        Annotation("dark brown", "COLOR", (2, 11)),
        Annotation("fox", "ANIMAL", (13, 15))]
    annotated_document_2 = AnnotatedDocument(content, annotations=annotations)

    content = b"The fox that was dark brown jumps over a dog that was lazy."
    annotations = [
        Annotation("fox", "ANIMAL", (4, 6)),
        Annotation("dark brown", "COLOR", (17, 26)),
        Annotation("dog", "ANIMAL", (41, 43))]
    annotated_document_3 = AnnotatedDocument(content, annotations=annotations)

    data = [annotated_document_1, annotated_document_2, annotated_document_3]

    # Train.
    model = SpaCyStatisticalNER()
    model.fit(data, num_epochs=5)

    # Predict. Works!
    content = b"The dark brown fox."
    document = Document(content)

    ann = model.transform([document])

    # 2 annotations, dark brown and fox
    assert_equal(len(ann[0].annotations), 2)
    assert_equal(ann[0].annotations[0].text, "dark brown")
    assert_equal(ann[0].annotations[0].offset, (4, 13))
    assert_equal(ann[0].annotations[0].label, "COLOR")
    assert_equal(ann[0].annotations[1].text, "fox")
    assert_equal(ann[0].annotations[1].offset, (15, 17))
    assert_equal(ann[0].annotations[1].label, "ANIMAL")

    entities = model.extract([document])
    assert_equal(len(entities[0]), 2)
