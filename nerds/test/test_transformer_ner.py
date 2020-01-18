import warnings
warnings.filterwarnings("ignore")

from nose.tools import assert_equal, assert_true

from nerds.models import TransformerNER
from nerds.utils import load_data_and_labels

import numpy as np
import shutil


def test_bert_ner():
    X, y = load_data_and_labels("nerds/test/data/example.iob")
    model = TransformerNER(model_dir="nerds/test/data/models", max_iter=1)
    model.fit(X, y)
    model.save()
    model_r = model.load()
    y_pred = model_r.predict(X)
    assert_equal(len(y), len(y_pred), "Number of labels and predictions must be equal")
    assert_equal(len(y[0]), len(y_pred[0]), "Size of first Label and prediction must be equal")
    # shutil.rmtree("nerds/test/data/models")
