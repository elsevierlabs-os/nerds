from nose.tools import assert_equal, assert_raises

from nerds.core.model.config.base import (
    NERModelConfiguration, load_config, validate_config)
from nerds.core.model.config.error import ConfigurationError


def test_ner_model_config():
    ner_config = NERModelConfiguration("nerds/test/data/config/sample.yaml")
    expected = {
        "model_config": {
            "crf": {
                "c1": 0.1,
                "c2": 0.1,
                "max_iterations": 100
            }
        },
        "ensemble_config": {
            "vote": "majority"
        },
        "entity_label": "all"
    }
    assert_equal(ner_config.config, expected)


def test_ner_model_config_missing_key():
    config = load_config("nerds/test/data/config/sample_error.yaml")
    assert_raises(ConfigurationError, validate_config, config)
