import random
import shutil

from nerds.core.model.ner.crf import CRF
from nerds.core.model.ner.spacy import SpaCyStatisticalNER
from nerds.core.model.ner.bilstm import BidirectionalLSTM
from nerds.core.model.ner.ensemble import (
    NERModelEnsembleMajorityVote, NERModelEnsemblePooling)
from nerds.core.model.eval.score import calculate_precision_recall_f1score

from read_data import data_to_annotated_docs

X = data_to_annotated_docs()
print("Loaded data: ", len(X), "data points")
random.Random(42).shuffle(X)

entity_names = ['art', 'org', 'geo', 'nat', 'gpe', 'per', 'eve', 'tim']
print("All labels: ", entity_names)

train_test_split = 0.8
train_X = X[:int(0.8 * len(X))]
test_X = X[int(0.8 * len(X)):]
print("Training: ", len(train_X))
print("Training: ", len(test_X))


def test_CRF():
    crf_model = CRF()
    crf_model.fit(train_X[:5000])

    X_pred = crf_model.transform(test_X)

    for l in entity_names:
        p, r, f = calculate_precision_recall_f1score(X_pred,
                                                     test_X,
                                                     entity_label=l)
        print("Label: ", l, p, r, f)

    # Save for ensemble usage to avoid training again.
    crf_model.save("tmp")


def test_spacy():
    spacy_model = SpaCyStatisticalNER()
    # Using the entire dataset will make Spacy die!
    spacy_model.fit(train_X[:5000])

    X_pred = spacy_model.transform(test_X)

    for l in entity_names:
        p, r, f = calculate_precision_recall_f1score(X_pred,
                                                     test_X,
                                                     entity_label=l)
        print("Label: ", l, p, r, f)

    # Save for ensemble usage to avoid training again.
    spacy_model.save("tmp")


def test_LSTM():
    lstm_model = BidirectionalLSTM()
    lstm_model.fit(train_X[:5000])

    X_pred = lstm_model.transform(test_X)

    for l in entity_names:
        p, r, f = calculate_precision_recall_f1score(X_pred,
                                                     test_X,
                                                     entity_label=l)
        print("Label: ", l, p, r, f)

    # Save for ensemble usage to avoid training again.
    lstm_model.save("tmp")


def test_ensembles():
    lstm_model = BidirectionalLSTM()
    lstm_model.load("tmp")
    spacy_model = SpaCyStatisticalNER()
    spacy_model.load("tmp")
    crf_model = CRF()
    crf_model.load("tmp")

    models = [lstm_model, crf_model, spacy_model]
    ens1 = NERModelEnsembleMajorityVote(models)
    ens2 = NERModelEnsemblePooling(models)

    X_pred_1 = ens1.transform(test_X)
    print("Majority Vote: \n")
    for l in entity_names:
        p, r, f = calculate_precision_recall_f1score(X_pred_1,
                                                     test_X,
                                                     entity_label=l)
        print("Label: ", l, p, r, f)

    X_pred_2 = ens2.transform(test_X)
    print("Pooling: \n")
    for l in entity_names:
        p, r, f = calculate_precision_recall_f1score(X_pred_2,
                                                     test_X,
                                                     entity_label=l)
        print("Label: ", l, p, r, f)


test_LSTM()
test_CRF()
test_spacy()
test_ensembles()

# Clean-up the model dirs.
shutil.rmtree("tmp/")
