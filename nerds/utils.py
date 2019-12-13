import anago
import itertools
import logging


def get_logger(log_level="DEBUG"):
    # TODO: The log level should be adjusted by some kind of configuration
    # file, e.g. the dev build should have DEBUG, while the release build
    # should have "WARN" or higher.
    f = "%(levelname)s %(asctime)s %(module)s %(filename)s: %(message)s"
    logging.basicConfig(format=f)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    return logger


def load_data_and_labels(filepath):
    """ Wrapper to expose anago's load_data_and_labels. Built here as
        a wrapper because users of non-neural models are not expected
        to be familiar with Anago.

        Args:
            filepath (str): path to the file in BIO format to be loaded.
        
        Returns:
            x (list(str)): list of tokens.
            y (list(str)): list of tags.
    """
    return anago.utils.load_data_and_labels(filepath)


def flatten_list(xs, strip_prefix=True, capture_lengths=False):
    """ Flatten label or predictions from list(list(str)) to list(str).
        Flattened list can be input to scikit-learn's standard functions
        to compute various metrics.

        Args:
            xs (list(list(str))): list of list of tags (inner list is sentence).
            strip_prefix (bool): if True, remove leading I- and B-, else retain.

        Returns:
            xs_flat list(str): the flattened list.
            xs_lengths list(int) or None: a list of lengths of the inner list(str)
                of the input xs.
    """
    def strip_bio_prefix(label):
        return label.split('-')[-1]

    if strip_prefix:
        return [strip_bio_prefix(x) for x in itertools.chain.from_iterable(xs)]
    else:
        return [x for x in itertools.chain.from_iterable(xs)]


def compute_list_lengths(xs):
    """ Convenience method to return a list of ints representing lengths of 
        inner lists in xs.

        Args:
            xs (list(list(str))): list of list of tags.
        
        Returns:
            xs_lengths (list(int)): list of lengths of inner list.
    """
    return [len(x) for x in xs]


def unflatten_list(xs_flat, xs_lengths):
    """ Reverse operation of flatten_list. Using the flattened list and the list
        of list lengths of the inner list, reconstructs original list(list(str)).

        Args:
            xs_flat list(str): the flattened list.
            xs_lengths list(int): list of inner list to group by.

        Returns:
            xs_unflat list(list(str)): original list of list(list(str))
    """
    xs_unflat = []
    start = 0
    for l in xs_lengths:
        end = start + l
        xs_unflat.append(xs_flat[start:end])
        start = end
    return xs_unflat


def tokens_to_spans(tokens, tags, allow_multiword_spans=True):
    """ Convert from tokens-tags format to sentence-span format. Some NERs
        use the sentence-span format, so we need to transform back and forth.

        Args:
            tokens (list(str)): list of tokens representing single sentence.
            tags (list(str)): list of tags in BIO format.
            allow_multiword_spans (bool): if True, offsets for consecutive 
                tokens of the same entity type are merged into a single span, 
                otherwise tokens are reported as individual spans.

        Returns:
            sentence (str): the sentence as a string.
            spans (list((int, int, str))): list of spans as a 3-tuple of start
                position, end position, and entity type. Note that end position
                is 1 beyond the actual ending position of the token.
    """
    spans = []
    curr, start, end, ent_cls = 0, None, None, None
    sentence = " ".join(tokens)
    if allow_multiword_spans:
        for token, tag in zip(tokens, tags):
            if tag == "O":
                if ent_cls is not None:
                    spans.append((start, end, ent_cls))
                    start, end, ent_cls = None, None, None
            elif tag.startswith("B-"):
                ent_cls = tag.split("-")[1]
                start = curr
                end = curr + len(token)
            else: # I-xxx
                end += len(token) + 1
            # advance curr
            curr += len(token) + 1
        
        # handle remaining span
        if ent_cls is not None:
            spans.append((start, end, ent_cls))
    else:
        for token, tag in zip(tokens, tags):
            if tag.startswith("B-") or tag.startswith("I-"):
                ent_cls = tag.split("-")[1]
                start = curr
                end = curr + len(token)
                spans.append((start, end, ent_cls))
            curr += len(token) + 1

    return sentence, spans


def spans_to_tokens(sentence, spans, spacy_lm, spans_are_multiword=True):
    """ Convert from sentence-spans format to tokens-tags format. Some NERs 
        use the sentence-spans format, so we need to transform back and forth.

        Args:
            sentence (str): the sentence as a string.
            spans (list((int, int, str))): list of spans as a 3-tuple of
                start_position, end_position, and entity_type. Note that end
                position is 1 beyond actual end position of the token.
            spacy_lm: we use SpaCy EN language model to tokenizing the 
                sentence to generate list of tokens.
            spans_are_multiword (bool): if True, indicates that spans can
                be multi-word spans), so consecutive entries of the same class 
                should be transformed, ie. (B-x, B-x) should become (B-x, I-x).

        Returns:
            tokens (list(str)): list of tokens in sentence
            tags (list(str)): list of tags in BIO format.
    """
    tokens, tags = [], []
    curr_start, curr_end = 0, 0
    for t in spacy_lm(sentence):
        tokens.append(t.text)
        curr_end = curr_start + len(t.text)
        is_annotated = False
        for span_start, span_end, span_cls in spans:
            if curr_start == span_start:
                tags.append("B-" + span_cls)
                is_annotated = True
                break
            elif curr_start > span_start and curr_end <= span_end:
                tags.append("I-" + span_cls)
                is_annotated = True
                break
            else:
                continue
        if not is_annotated:
            tags.append("O")

        curr_start += len(t.text) + 1

    # handle consecutive class labels if spans were single word spans
    if not spans_are_multiword:
        prev_tag, merged_tags = None, []
        for tag in tags:
            if prev_tag is None or prev_tag != tag:
                merged_tags.append(tag)
            else:
                merged_tags.append(tag.replace("B-", "I-"))
            prev_tag = tag
        tags = merged_tags

    return tokens, tags

