import csv
import os
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from nerds.models import (
    DictionaryNER, SpacyNER, CrfNER, BiLstmCrfNER, 
    ElmoNER, FlairNER, BertNER, TransformerNER, 
    EnsembleNER
)
from nerds.utils import *


def convert_to_iob_format(input_file, output_file):
    num_written = 0
    fout = open(output_file, "w")
    with open(input_file, "r", encoding="iso-8859-1") as fin:
        csv_reader = csv.reader(fin, delimiter=',', quotechar='"')
        # skip header
        next(csv_reader)
        for line in csv_reader:
            sid, token, pos, tag = line
            if num_written > 0:
                if len(sid) != 0:
                    # end of sentence marker
                    fout.write("\n")
            fout.write("\t".join([token, tag]) + "\n")
            num_written += 1

    fout.write("\n")
    fout.close()


# convert GMB dataset to our standard IOB format
if not os.path.exists("train.iob"):
    convert_to_iob_format("train.csv", "train.iob")

# these are our entities
entity_labels = ["art", "eve", "geo", "gpe", "nat", "org", "per", "tim"]

# make model directory to store our models
if not os.path.exists("models"):
    os.makedirs("models")

# read IOB file 
data, labels = load_data_and_labels("train.iob", encoding="iso-8859-1")
# optional: restrict dataset to 5000 sentences
# data_s, labels_s = shuffle(data, labels, random_state=42)
# data = data_s
# labels = labels_s
print(len(data), len(labels))

# split into train and test set
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, 
    test_size=0.3, random_state=42)
print(len(xtrain), len(ytrain), len(xtest), len(ytest))

# train and test the dictionary NER
model = DictionaryNER()
model.fit(xtrain, ytrain)
model.save("models/dict_model")
trained_model = model.load("models/dict_model")
ypred = trained_model.predict(xtest)
print(classification_report(flatten_list(ytest, strip_prefix=True),
                            flatten_list(ypred, strip_prefix=True),
                            labels=entity_labels))

# train and test the CRF NER
model = CrfNER()
model.fit(xtrain, ytrain)
model.save("models/crf_model")
trained_model = model.load("models/crf_model")
ypred = trained_model.predict(xtest)
print(classification_report(flatten_list(ytest, strip_prefix=True),
                            flatten_list(ypred, strip_prefix=True),
                            labels=entity_labels))

# train and test the SpaCy NER
model = SpacyNER()
model.fit(xtrain, ytrain)
model.save("models/spacy_model")
trained_model = model.load("models/spacy_model")
ypred = trained_model.predict(xtest)
print(classification_report(flatten_list(ytest, strip_prefix=True),
                            flatten_list(ypred, strip_prefix=True),
                            labels=entity_labels))

# train and test the BiLSTM-CRF NER
model = BiLstmCrfNER()
model.fit(xtrain, ytrain)
model.save("models/bilstm_model")
trained_model = model.load("models/bilstm_model")
ypred = trained_model.predict(xtest)
print(classification_report(flatten_list(ytest, strip_prefix=True),
                            flatten_list(ypred, strip_prefix=True),
                            labels=entity_labels))

# train and test the ELMo NER
if os.path.exists("glove.6B.100d.txt"):
    model = ElmoNER()
    model.fit(xtrain, ytrain)
    model.save("models/elmo_model")
    trained_model = model.load("models/elmo_model")
    ypred = trained_model.predict(xtest)
    print(classification_report(flatten_list(ytest, strip_prefix=True),
                                flatten_list(ypred, strip_prefix=True),
                                labels=entity_labels))

# train and test the FLAIR NER
model = FlairNER("models/flair_model")
model.fit(xtrain, ytrain)
model.save("models/flair_model")
trained_model = model.load("models/flair_model")
ypred = trained_model.predict(xtest)
print(classification_report(flatten_list(ytest, strip_prefix=True),
                            flatten_list(ypred, strip_prefix=True),
                            labels=entity_labels))

# train and test the BERT NER
model = BertNER(padding_tag="X")
model.fit(xtrain, ytrain)
model.save("models/bert_model")
trained_model = model.load("models/bert_model")
ypred = trained_model.predict(xtest)
ytest, ypred = align_labels_and_predictions(ypred, ytest, padding_tag="X")
print(classification_report(flatten_list(ytest, strip_prefix=True),
                            flatten_list(ypred, strip_prefix=True),
                            labels=entity_labels))

# train and test Transformer NER
model = TransformerNER(
    model_dir="models/transformer_model",
    padding_tag="X")
model.fit(xtrain, ytrain)
model.save()
trained_model = model.load()
ypred = trained_model.predict(xtest)
ytest, ypred = align_labels_and_predictions(ypred, ytest, padding_tag="X")
print(classification_report(flatten_list(ytest, strip_prefix=True),
                            flatten_list(ypred, strip_prefix=True),
                            labels=entity_labels))

# create and test an ensemble
dict_model = DictionaryNER()
dict_model.load("models/dict_model")
crf_model = CrfNER()
crf_model.load("models/crf_model")
spacy_model = SpacyNER()
spacy_model.load("models/spacy_model")
bilstm_model = BiLstmCrfNER()
bilstm_model.load("models/bilstm_model")
estimators = [
    ("dict_model", dict_model),
    ("crf_model", crf_model),
    ("spacy_model", spacy_model),
    ("bilstm_model", bilstm_model)
]
model = EnsembleNER(estimators=estimators, is_pretrained=True)
ypred = model.predict(xtest)
print(classification_report(flatten_list(ytest, strip_prefix=True),
                            flatten_list(ypred, strip_prefix=True),
                            labels=entity_labels))

# clean up
shutil.rmtree("models")
os.remove("train.iob")
os.remove("glove.6B.100d.txt")