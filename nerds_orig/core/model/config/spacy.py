from nerds.core.model.config.base import NERModelConfiguration
from nerds.core.model.ner.spacy import SpaCyStatisticalNER


class SpaCyStatisticalNERConfiguration(NERModelConfiguration):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = SpaCyStatisticalNER(self.entity_label)
