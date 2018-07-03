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
