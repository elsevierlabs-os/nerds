import nltk
import regex as re
import spacy

# python -m spacy download en
NLP = spacy.load('en')

TOKENIZATION_REGEXP = re.compile("([\pL]+|[\d]+|[^\pL])")


def document_to_sentences(string):
    """ Given a document (free-form string), we use SpaCy to give us sentences.

        Args:
            string (str): The input string, corresponding to a document.

        Returns:
            list(str): List of segmented sentences.
    """

    return [sent.text for sent in NLP(string.strip()).sents]


def sentence_to_tokens(string, method="regexp"):
    """ Given a sentence (free-form string), we tokenize it according to a
        method. Two methods are currently supported: (i) "regexp", which
        tokenizes on every word, symbol and number and (ii) "statistical",
        which uses SpaCy's trained statistical tokenizer.

        Args:
            string (str): The input sentence.
            method (str, optional): Either "regexp" or "statistical".
                Defaults to "regexp"

        Returns:
            list(str): List of tokens.

        Raises:
            TypeError: If the tokenization method is not supported.
    """
    if method not in ("regexp", "statistical"):
        raise TypeError("Tokenization method is not supported.")
    if method == "regexp":
        return [token.strip() for token in TOKENIZATION_REGEXP.split(string)
                if len(token.strip()) > 0]
    return [token.text for token in NLP(string.strip())]


def tokens_to_pos_tags(tokenized_sentence):
    """ Given a tokenized sentence, we use the NLTK tagger to give us POS tags.

        Args:
            tokenized_sentence (list(str)): The input sentence
                post-tokenization.

        Returns:
            list(str): List of POS tags.
    """

    return [tag for _, tag in nltk.pos_tag(tokenized_sentence)]
