import yaml
from sklearn.base import BaseEstimator, ClassifierMixin

from nerds.core.model.config.error import ConfigurationError
from nerds.util.logging import get_logger


log = get_logger()


CONFIG_KEYS = (
    "model_config",
    "ensemble_config",
    "entity_label")


def load_config(config_path):
    with open(config_path, "r") as fp:
        return yaml.load(fp.read().strip())


def validate_config(config):
    for key in CONFIG_KEYS:
        if key not in config.keys():
            msg = "Configuration missing key: {}".format(key)
            log.error(msg)
            raise ConfigurationError(msg)


class NERModelConfiguration(BaseEstimator, ClassifierMixin):
    """ Wrapper class that initializes a NER model with a configuration.

        It accepts as input the path to a YAML file that must contain the
        following keys: `model_config`, `ensemble_config`, and `entity_label`.

        `model_config` contains model-specific configurations i.e. the
        hyperparameters that initialize a model. Example:
        ```
        model_config:
            crf:
                c1: 0.1
                c2: 0.1
                max_iterations: 100
            spacy:
                num_epochs: 20
                dropout: 0.1
        ```

        `ensemble_config` contains the voting mechanism of the ensembler.
        Example:
        ```
        ensemble_config:
            vote: majority
        ```

        `entity_label` is the entity for which we're training the models,
        or "all" if we want to train for all available entities in the dataset.
        Example:
        ```
        entity_label: all
        ```

        Attributes:
            config_path (str): The path to the config file that initializes the
                NER models.
    """

    def __init__(self, config_path):
        config = load_config(config_path)
        validate_config(config)
        if config["entity_label"] != "all":
            self.entity_label = config["entity_label"]
        else:
            self.entity_label = None
        self.model_config = config["model_config"]
        self.ensemble_config = config["ensemble_config"]["vote"]
        self.model = None  # To be added in subclass
        self.config = config
        self.config_path = config_path

    def fit(self, X, y=None):
        """ Trains the NER model this configuration wrapper contains.
            The input is a list of `AnnotatedDocument` instances.
        """
        if self.model is None:
            raise NotImplementedError("Must extend this class with a model.")
        return self.model.fit(X, y, **self.config[self.model.key])

    def transform(self, X, y=None):
        """ Uses the NER model this wrapper contains to annotate the list of
            `Document` objects that are provided as input and returns a list
            of `AnnotatedDocument` objects.
        """
        if self.model is None:
            raise NotImplementedError("Must extend this class with a model.")
        return self.model.transform(X, y)

    def extract(self, X, y=None):
        """ Returns a list of entities, extracted from annotated documents. """
        if self.model is None:
            raise NotImplementedError("Must extend this class with a model.")
        return self.model.extract(X, y)
