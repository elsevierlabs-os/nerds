from nose.tools import assert_equal, assert_in, assert_true

from nerds.core.model.input.document import Document
from nerds.core.model.ner.dictionary import ExactMatchDictionaryNER
from nerds.core.model.ner.dictionary import ExactMatchMultiClassDictionaryNER


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

def test_ExactMatchMultiClassDictionaryNER():
    document = Document(b"""
    In this study , we have used the polymerase chain reaction ( PCR ) with nested 
    primers to analyze X-inactivation patterns of the HUMARA loci in purified eosinophils 
    from female patients with eosinophilia .
    """)
    ner = ExactMatchMultiClassDictionaryNER(
        "nerds/test/data/dictionary/biodictionary.txt")
    annotated = ner.transform([document])
    expected_labels = ["DNA", "cell-type"]
    for i, annotation in enumerate(annotated[0].annotations):
        pred_text = annotation.text
        pred_offsets = annotation.offset
        label_text = document.plain_text_[pred_offsets[0]:pred_offsets[1]]
        assert_equal(pred_text, label_text, 
            "predicted {:s} != label {:s}".format(pred_text, label_text))
        assert_equal(annotation.label, expected_labels[i])

    
