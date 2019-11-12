from nose.tools import assert_equal

from nerds.core.model.input import Annotation, AnnotatedDocument, Document


def test_plain_text():
    document = Document(b"SIMPLE TEXT    HERE!!!")
    no_norm_text = document.plain_text_
    assert_equal(no_norm_text, "SIMPLE TEXT    HERE!!!")


def test_annotated_text():
    # Normal case: Entity in text.
    content = b"I love the smell of pizza in the morning."
    annotations = [
        Annotation("pizza", "FOOD", (20, 24))
    ]
    annotated_document = AnnotatedDocument(content, annotations=annotations)
    assert_equal(
        annotated_document.annotated_text_,
        "I love the smell of FOOD[pizza] in the morning.")

    # Special case: Text ends with a label (ANIMAL[dog])
    content = b"The quick brown fox jumps over the lazy dog."
    annotations = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42))
    ]
    annotated_document = AnnotatedDocument(content, annotations=annotations)
    assert_equal(
        annotated_document.annotated_text_,
        "The quick COLOR[brown] ANIMAL[fox] jumps over the lazy ANIMAL[dog].")
