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

