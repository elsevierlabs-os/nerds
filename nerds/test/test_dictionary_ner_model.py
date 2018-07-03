from nose.tools import assert_equal, assert_in

from nerds.core.model.input.document import Document
from nerds.core.model.ner.dictionary import ExactMatchDictionaryNER


def test_ExactMatchDictionaryNER():
    document = Document(b"""
    There are many publishers in the world, like
     Elsevier, Springer and also Wiley""")

    dictionary_ner = ExactMatchDictionaryNER(
        "nerds/test/data/dictionary/orgdictionary.txt", "ORGANIZATION")
    annotated = dictionary_ner.transform([document])

    annotations = annotated[0].annotations

    assert_equal(
        3, len(annotations), "Must have matched the three publishers.")

    unique_annotations = set([ann.text for ann in annotations])

    assert_in("Elsevier", unique_annotations)
    assert_in("Springer", unique_annotations)
    assert_in("Wiley", unique_annotations)
