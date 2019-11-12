from nose.tools import assert_equal

from nerds.core.model.config.ensemble import (
    NERModelEnsembleConfiguration, _get_ensembler_by_voting_method)
from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import AnnotatedDocument, Document
from nerds.core.model.ner.ensemble import (
    NERModelEnsemblePooling,
    NERModelEnsembleMajorityVote,
    NERModelEnsembleWeightedVote)


def test_get_ensembler_by_voting_method():
    ens_1 = _get_ensembler_by_voting_method("pooling")
    ens_1_expected = NERModelEnsemblePooling([])
    assert_equal(type(ens_1), type(ens_1_expected))

    ens_2 = _get_ensembler_by_voting_method("majority")
    ens_2_expected = NERModelEnsembleMajorityVote([])
    assert_equal(type(ens_2), type(ens_2_expected))

    ens_3 = _get_ensembler_by_voting_method("weighted")
    ens_3_expected = NERModelEnsembleWeightedVote([])
    assert_equal(type(ens_3), type(ens_3_expected))


def test_ner_ensemble_configuration():
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

    # In that config file we have params for a CRF with c1 = 0.1 and c2 = 0.1.
    ner_ensemble_config = NERModelEnsembleConfiguration(
        "nerds/test/data/config/sample.yaml")
    ner_ensemble_config.fit([
        annotated_document_1,
        annotated_document_2,
        annotated_document_3])

    # Predict. Works!
    content = b"The quick brown fox."
    document = Document(content)

    ann = ner_ensemble_config.transform([document])

    # 2 annotations, brown and fox
    assert_equal(len(ann[0].annotations), 2)
    assert_equal(ann[0].annotations[0].text, "brown")
    assert_equal(ann[0].annotations[0].label, "COLOR")
    assert_equal(ann[0].annotations[0].offset, (10, 14))
    assert_equal(ann[0].annotations[1].text, "fox")
    assert_equal(ann[0].annotations[1].label, "ANIMAL")
    assert_equal(ann[0].annotations[1].offset, (16, 18))
