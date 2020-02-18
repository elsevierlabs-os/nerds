from nerds.core.model.config.base import NERModelConfiguration
from nerds.core.model.ner.bilstm import BidirectionalLSTM


class BidirectionalLSTMConfiguration(NERModelConfiguration):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model = BidirectionalLSTM(self.entity_label)
