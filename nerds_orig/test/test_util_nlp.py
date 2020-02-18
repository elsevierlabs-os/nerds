from nose.tools import assert_equal

from nerds.util.nlp import (
    document_to_sentences, sentence_to_tokens,
    tokens_to_pos_tags)


def test_document_to_sentences():

    document = "This is a document. There are two sentences."
    split_sentences = document_to_sentences(document)

    assert_equal(2, len(split_sentences))
    assert_equal("This is a document.", split_sentences[0])
    assert_equal("There are two sentences.", split_sentences[1])


def test_sentence_to_tokens():

    sentence = "This is a sentence, with many (tokens)."
    split_tokens = sentence_to_tokens(sentence)

    assert_equal(11, len(split_tokens))

    assert_equal(["This",
                  "is",
                  "a",
                  "sentence",
                  ",",
                  "with",
                  "many",
                  "(",
                  "tokens",
                  ")",
                  "."], split_tokens)


def test_tokens_to_pos_tags():

    sentence = "The cat (feline) eats a mouse."
    split_tokens = sentence_to_tokens(sentence)
    pos_tags = tokens_to_pos_tags(split_tokens)

    assert_equal(['DT', 'NN', '(', 'NN', ')', 'VBZ', 'DT', 'NN', '.'],
                 pos_tags)
