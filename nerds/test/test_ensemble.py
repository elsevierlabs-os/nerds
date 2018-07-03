from nose.tools import assert_equal

from nerds.core.model.input.annotation import Annotation
from nerds.core.model.ner.base import NERModel
from nerds.core.model.ner.ensemble import (
    NERModelEnsemblePooling, NERModelEnsembleMajorityVote,
    NERModelEnsembleWeightedVote)


def test_ensemble_pooling():
    ensemble = NERModelEnsemblePooling(
        [NERModel(), NERModel(), NERModel()])

    x1 = Annotation("pizza", "FOOD", (1, 6))
    x2 = Annotation("milk", "FOOD", (7, 11))
    x3 = Annotation("cupcake", "FOOD", (12, 19))
    x4 = Annotation("kebab", "FOOD", (20, 25))
    x5 = Annotation("pie", "FOOD", (29, 32))
    x6 = Annotation("cheese", "FOOD", (35, 41))

    entity_matrix = [
        [x1, x2, x4],
        [x1, x3, x5],
        [x1, x2, x4, x6]
    ]

    assert_equal(ensemble.vote(entity_matrix), [x1, x2, x3, x4, x5, x6])


def test_ensemble_majority_vote():
    ensemble = NERModelEnsembleMajorityVote(
        [NERModel(), NERModel(), NERModel()])

    x1 = Annotation("pizza", "FOOD", (1, 6))
    x2 = Annotation("milk", "FOOD", (7, 11))
    x3 = Annotation("cupcake", "FOOD", (12, 19))
    x4 = Annotation("kebab", "FOOD", (20, 25))
    x5 = Annotation("pie", "FOOD", (29, 32))
    x6 = Annotation("cheese", "FOOD", (35, 41))

    entity_matrix = [
        [x1, x2, x4],
        [x1, x3, x5],
        [x1, x2, x4, x6]
    ]

    # Majority vote: 2 out of 3 classifiers voted for x1, x2 and x4.
    assert_equal(ensemble.vote(entity_matrix), [x1, x2, x4])


def test_ensemble_weighted_vote():
    ensemble = NERModelEnsembleWeightedVote(
        [NERModel(), NERModel(), NERModel()])

    ensemble.confidence_scores = [0.4, 0.7, 0.3]

    x1 = Annotation("pizza", "FOOD", (1, 6))
    x2 = Annotation("milk", "FOOD", (7, 11))
    x3 = Annotation("cupcake", "FOOD", (12, 19))
    x4 = Annotation("kebab", "FOOD", (20, 25))
    x5 = Annotation("pie", "FOOD", (29, 32))
    x6 = Annotation("cheese", "FOOD", (35, 41))

    entity_matrix = [
        [x1, x2, x4],
        [x1, x3, x5],
        [x1, x2, x4, x6]
    ]

    # Unlike the majority vote, here we expect to see x3 and x5 in the
    # annotations, because they come from a classifier of significantly
    # higher confidence.
    assert_equal(ensemble.vote(entity_matrix), [x1, x2, x3, x4, x5])
