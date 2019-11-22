import os
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from nerds.models import (
    DictionaryNER, SpacyNER, CrfNER, BiLstmCrfNER, ElmoNER, EnsembleNER
)
from nerds.utils import *

# these are our entities
entity_labels = ["cell_line", "cell_type", "protein", "DNA", "RNA"]

# load data
xtrain, ytrain = load_data_and_labels("data/train/Genia4ERtask1.iob2")
xtest, ytest = load_data_and_labels("data/test/Genia4EReval1.iob2")
print(len(xtrain), len(ytrain), len(xtest), len(xtest))

# make model directory to store our models
if not os.path.exists("models"):
    os.makedirs("models")

# # train and test the Dictionary NER
# model = DictionaryNER()
# model.fit(xtrain, ytrain)
# model.save("models/dict_model")
# trained_model = model.load("models/dict_model")
# ypred = trained_model.predict(xtest)
# print(classification_report(flatten_list(ytest, strip_prefix=True),
#                             flatten_list(ypred, strip_prefix=True),
#                             labels=entity_labels))

# # train and test the CRF NER
# model = CrfNER()
# model.fit(xtrain, ytrain)
# model.save("models/crf_model")
# trained_model = model.load("models/crf_model")
# ypred = trained_model.predict(xtest)
# print(classification_report(flatten_list(ytest, strip_prefix=True),
#                             flatten_list(ypred, strip_prefix=True),
#                             labels=entity_labels))

# # train and test the SpaCy NER
# model = SpacyNER()
# model.fit(xtrain, ytrain)
# model.save("models/spacy_model")
# trained_model = model.load("models/spacy_model")
# ypred = trained_model.predict(xtest)
# print(classification_report(flatten_list(ytest, strip_prefix=True),
#                             flatten_list(ypred, strip_prefix=True),
#                             labels=entity_labels))

# # train and test the BiLSTM-CRF NER
# model = BiLstmCrfNER()
# model.fit(xtrain, ytrain)
# model.save("models/bilstm_model")
# trained_model = model.load("models/bilstm_model")
# ypred = trained_model.predict(xtest)
# print(classification_report(flatten_list(ytest, strip_prefix=True),
#                             flatten_list(ypred, strip_prefix=True),
#                             labels=entity_labels))

# # train and test the ELMo NER
# if os.path.exists("glove.6B.100d.txt"):
#     model = ElmoNER()
#     model.fit(xtrain, ytrain)
#     model.save("models/elmo_model")
#     trained_model = model.load("models/elmo_model")
#     ypred = trained_model.predict(xtest)
#     print(classification_report(flatten_list(ytest, strip_prefix=True),
#                                 flatten_list(ypred, strip_prefix=True),
#                                 labels=entity_labels))

# create and test an ensemble
dict_model = DictionaryNER()
dict_model.load("models/dict_model")
crf_model = CrfNER()
crf_model.load("models/crf_model")
spacy_model = SpacyNER()
spacy_model.load("models/spacy_model")
bilstm_model = BiLstmCrfNER()
bilstm_model.load("models/bilstm_model")
model = EnsembleNER()
model.fit(xtrain, ytrain, 
    estimators=[
        (dict_model, {}),
        (crf_model, {}),
        (spacy_model, {}),
        (bilstm_model, {})
    ],
    is_pretrained=True)
ypred = model.predict(xtest)
print(classification_report(flatten_list(ytest, strip_prefix=True),
                            flatten_list(ypred, strip_prefix=True),
                            labels=entity_labels))

# # clean up
# shutil.rmtree("models")
# shutil.rmtree("data")
# os.remove("glove.6B.100d.txt")