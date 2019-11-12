from nerds.core.model.config.base import NERModelConfiguration
from nerds.core.model.ner.crf import CRF


class CRFConfiguration(NERModelConfiguration):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = CRF(self.entity_label)
