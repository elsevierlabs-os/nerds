from nose.tools import assert_equal, assert_true

from nerds.utils import *

import spacy

spacy_lm = spacy.load("en")

def test_load_data_and_labels():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    assert_true(len(X) == 2, "There should be 2 sentences in X")
    assert_equal(len(X), len(y), "There should be tags for 2 sentences in y")
    assert_equal(len(X[0]), len(y[0]), "Number of tokens should be equal to number of tags")


def test_flatten_and_unflatten_list():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    yflat = flatten_list(y, strip_prefix=True, capture_lengths=True)
    assert_equal(36, len(yflat), "There should be 36 tags in all")
    assert_equal(5, len([y for y in yflat if y == "PER"]), "There should be 5 PER tags")
    y_lengths = compute_list_lengths(y)
    y_unflat = unflatten_list(yflat, y_lengths)
    assert_equal(len(y), len(y_unflat), "Reconstructed y (y_unflat) should be identical to y")
    assert_equal(len(y[0]), len(y_unflat[0]), "Reconstructed y (y_unflat) should be identical to y")


def test_tokens_to_spans():
    data, labels = load_data_and_labels("nerds/test/data/example.iob")
    tokens, tags = data[0], labels[0]
    sentence, spans = tokens_to_spans(tokens, tags)
    assert_equal(
        "Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov . 29 .",
        sentence, "Sentence reconstruction is incorrect")
    assert_equal(3, len(spans), "Should be exactly 3 spans")
    assert_equal(0, spans[0][0], "spans[0].start should be 0")
    assert_equal(13, spans[0][1], "spans[0].end should be 13")
    assert_equal("PER", spans[0][2], "spans[0].cls should be PER")
    assert_equal(16, spans[1][0], "spans[1].start should be 16")
    assert_equal(28, spans[1][1], "spans[1].end should be 28")
    assert_equal("DATE", spans[1][2], "spans[1].cls should be DATE")
    assert_equal(78, spans[2][0], "spans[2].start should be 78")
    assert_equal(86, spans[2][1], "spans[2].end should be 86")
    assert_equal("DATE", spans[2][2], "spans[2].cls should be DATE")


def test_spans_to_tokens():
    sentence = "Mr . Vinken is chairman of Elsevier N . V . , the Dutch publishing group ."
    spans = [(0, 11, "PER"), (27, 43, "ORG"), (50, 55, "NORP")]
    tokens, tags = spans_to_tokens(sentence, spans, spacy_lm)
    # reference tokens and tags for comparison
    data, labels = load_data_and_labels("nerds/test/data/example.iob")
    ref_tokens, ref_tags = data[1], labels[1]
    assert_equal(len(tokens), len(ref_tokens), "Number of tokens should be identical")
    for token, ref_token in zip(tokens, ref_tokens):
        assert_equal(ref_token, token, "Tokens do not match. {:s} != {:s}".format(ref_token, token))
    assert_equal(len(tags), len(ref_tags), "Number of BIO tags should be identical")
    for tag, ref_tag in zip(tags, ref_tags):
        assert_equal(ref_tag, tag, "Tags do not match. {:s} != {:s}".format(ref_tag, tag))
