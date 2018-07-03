from nerds.core.model.config.base import NERModelConfiguration
from nerds.core.model.ner.bilstm import BidirectionalLSTM
from nerds.core.model.ner.crf import CRF
from nerds.core.model.ner.ensemble import (
    NERModelEnsemblePooling,
    NERModelEnsembleMajorityVote,
    NERModelEnsembleWeightedVote)
from nerds.core.model.ner.spacy import SpaCyStatisticalNER

""" The following lines define a mapping from the configuration keys for each
    model to their corresponding classes.
    There is a similar mapping in nerds.core.model.ner.ensemble that we use
    for persisting and loading models to and from the disk.
"""

# If a new model is supported, add it here.
SUPPORTED_MODELS = ("bilstm", "crf", "spacy")
# Maps the model names to their respective classes (useful for instantiation).
MODEL_CLASS_MAPPING = {
    "bilstm": BidirectionalLSTM,
    "crf": CRF,
    "spacy": SpaCyStatisticalNER
}


def _get_ensembler_by_voting_method(method):
    if method == "pooling":
        return NERModelEnsemblePooling([])
    if method == "majority":
        return NERModelEnsembleMajorityVote([])
    if method == "weighted":
        return NERModelEnsembleWeightedVote([])


class NERModelEnsembleConfiguration(NERModelConfiguration):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = _get_ensembler_by_voting_method(self.ensemble_config)
        self.model.models = [
            MODEL_CLASS_MAPPING[key](self.entity_label)
            for key in self.model_config.keys()
        ]

    def fit(self, X, y=None):
        return self.model.fit(X, y, self.model_config)
