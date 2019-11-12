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
            filepath (str): path to the file in IOB format to be loaded.
        
        Returns:
            x (list(str)): list of tokens.
            y (list(str)): list of tags.
    """
    return anago.utils.load_data_and_labels(filepath)


def flatten_lol(xs, strip_prefix=True):
    """ Flatten label or predictions from list(list(str)) to list(str).
        Flattened list can be input to scikit-learn's standard functions
        to compute various metrics.

        Args:
            xs (list(list(str))): list of list of tags (inner list is sentence).
            strip_prefix (bool): if True, remove leading I- and B-, else retain.
    """
    def strip_iob_prefix(label):
        return label.split('-')[-1]
    if strip_prefix:
        return [strip_iob_prefix(x) for x in itertools.chain.from_iterable(xs)]
    else:
        return [x for x in itertools.chain.from_iterable(xs)]


