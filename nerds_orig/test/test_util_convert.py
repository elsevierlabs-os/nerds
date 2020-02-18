from nose.tools import assert_equal

from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import Document
from nerds.core.model.input.document import AnnotatedDocument
from nerds.util.convert import (
    transform_annotated_documents_to_bio_format,
    transform_bio_tags_to_annotated_documents,
    split_annotated_document)


def test_transform_annotated_documents_to_bio_format():

    # Test no annotations.
    content = b"The quick brown fox jumps over the lazy dog."
    annotated_document = AnnotatedDocument(content, annotations=None)

    expected = (
        [['The', 'quick', 'brown', 'fox', 'jumps',
            'over', 'the', 'lazy', 'dog', '.']],
        [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']])

    transformed = transform_annotated_documents_to_bio_format([
        annotated_document])
    assert_equal(transformed, expected)

    annotations = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42))
    ]
    annotated_document = AnnotatedDocument(content, annotations=annotations)
    expected = (
        [['The', 'quick', 'brown', 'fox', 'jumps',
            'over', 'the', 'lazy', 'dog', '.']],
        [['O', 'O', 'B_COLOR', 'B_ANIMAL', 'O', 'O',
            'O', 'O', 'B_ANIMAL', 'O']])

    transformed = transform_annotated_documents_to_bio_format(
        [annotated_document])
    assert_equal(transformed, expected)


def test_transform_bio_tags_to_annotated_documents():

    # Test no annotations.
    content = b"The quick brown fox jumps over the lazy dog."
    documents = [Document(content)]

    tokens = [
        ['The', 'quick', 'brown', 'fox', 'jumps',
            'over', 'the', 'lazy', 'dog', '.']]
    bio = [['O', 'O', 'B_COLOR', 'B_ANIMAL', 'O', 'O',
            'O', 'O', 'B_ANIMAL', 'O']]

    transformed = transform_bio_tags_to_annotated_documents(
        tokens, bio, documents)

    assert_equal(len(transformed[0].annotations), 3)
    assert_equal(transformed[0].annotations[0].text, "brown")
    assert_equal(transformed[0].annotations[1].text, "fox")
    assert_equal(transformed[0].annotations[2].text, "dog")


def test_transform_bio_tags_to_annotated_documents_endswith_B():

    # Test no annotations.
    content = b"The quick brown fox jumps over the lazy dog"
    documents = [Document(content)]

    tokens = [
        ['The', 'quick', 'brown', 'fox', 'jumps',
            'over', 'the', 'lazy', 'dog']]
    bio = [['O', 'O', 'B_COLOR', 'B_ANIMAL', 'O', 'O',
            'O', 'O', 'B_ANIMAL']]

    transformed = transform_bio_tags_to_annotated_documents(
        tokens, bio, documents)

    assert_equal(len(transformed[0].annotations), 3)
    assert_equal(transformed[0].annotations[0].text, "brown")
    assert_equal(transformed[0].annotations[1].text, "fox")
    assert_equal(transformed[0].annotations[2].text, "dog")


def test_transform_bio_tags_to_annotated_documents_endswith_I():

    # Test no annotations.
    content = b"The quick brown fox jumps over the lazy German shepherd"
    documents = [Document(content)]

    tokens = [
        ['The', 'quick', 'brown', 'fox', 'jumps',
            'over', 'the', 'lazy', 'German', 'shepherd']]
    bio = [['O', 'O', 'B_COLOR', 'B_ANIMAL', 'O', 'O',
            'O', 'O', 'B_ANIMAL', 'I_ANIMAL']]

    transformed = transform_bio_tags_to_annotated_documents(
        tokens, bio, documents)

    assert_equal(len(transformed[0].annotations), 3)
    assert_equal(transformed[0].annotations[0].text, "brown")
    assert_equal(transformed[0].annotations[1].text, "fox")
    assert_equal(transformed[0].annotations[2].text, "German shepherd")


def test_split_annotated_document():
    content = (b"The quick brown fox jumps over the lazy dog. "
               b"Grumpy wizards make a toxic brew for the jovial queen.")
    annotations = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42)),
        Annotation("wizards", "PERSON", (52, 58)),
        Annotation("brew", "DRINK", (73, 76)),
        Annotation("queen", "PERSON", (93, 97))
    ]
    annotated_document = AnnotatedDocument(content, annotations)
    result = split_annotated_document(annotated_document)

    assert_equal(result[0].content,
                 b"The quick brown fox jumps over the lazy dog.")
    assert_equal(result[1].content,
                 b"Grumpy wizards make a toxic brew for the jovial queen.")

    expected_annotations_doc1 = [
        Annotation("brown", "COLOR", (10, 14)),
        Annotation("fox", "ANIMAL", (16, 18)),
        Annotation("dog", "ANIMAL", (40, 42))
    ]
    assert_equal(result[0].annotations, expected_annotations_doc1)

    expected_annotations_doc2 = [
        Annotation("wizards", "PERSON", (7, 13)),
        Annotation("brew", "DRINK", (28, 31)),
        Annotation("queen", "PERSON", (48, 52))
    ]
    assert_equal(result[1].annotations, expected_annotations_doc2)
