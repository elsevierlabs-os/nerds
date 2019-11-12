import json
import os.path

import sklearn_crfsuite
from sklearn.externals import joblib

from nerds.core.model.ner.base import NERModel
from nerds.util.convert import (
    transform_annotated_documents_to_bio_format,
    transform_bio_tags_to_annotated_documents)
from nerds.util.file import mkdir
from nerds.util.logging import get_logger
from nerds.util.nlp import tokens_to_pos_tags

log = get_logger()


class CRF(NERModel):

    def __init__(self, entity_label=None):
        super().__init__(entity_label)
        self.crf = None
        self.key = "crf"

    def fit(self, X, y=None, max_iterations=100, c1=0.1, c2=0.1):
        log.info("Generating features for {} samples...".format(len(X)))
        # Features and labels are useful for training.
        features, tokens, labels = self._preprocessor(X)

        log.info("Training the CRF...")
        # Configure training parameter.
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
            verbose=True)
        self.crf.fit(features, labels)

        return self

    def transform(self, X, y=None):
        log.info("Generating features for {} samples...".format(len(X)))
        features, tokens, labels = self._preprocessor(X)
        # Labels, of course, doesn't contain anything here.

        # Make predictions.
        predicted_labels = self.crf.predict(features)

        # Also need to make annotated documents.
        return transform_bio_tags_to_annotated_documents(
            tokens, predicted_labels, X)

    def _word_to_features(self, sent, i):
        """ Given a sentence, extract features from it for the CRF.

            TODO: Expose the parameters of this function somehow as
            global params.
        """

        word = sent[i][0]
        postag = sent[i][1]

        # As default, we have:
        # 1) A window size of 2, so 2 words before and 2 words after.
        # 2) Prefix and suffix of size 2.
        # 3) The word itself, lowercase.
        # 4) isupper, islower, begin with upper, isdigit.
        # 5) POS tags.
        features = {
            'bias': 1.0,
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[:3]': word[:3],
            'word[:2]': word[:2],
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.islower()': word.islower(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
        }
        if i > 1:
            word2 = sent[i - 2][0]
            postag2 = sent[i - 2][1]
            features.update({
                '-2:word[-3:]': word2[-3:],
                '-2:word[-2:]': word2[-2:],
                '-2:word[:3]': word2[:3],
                '-2:word[:2]': word2[:2],
                '-2:word.lower()': word2.lower(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.islower()': word2.islower(),
                '-2:word.isupper()': word2.isupper(),
                '-2:word.isdigit()': word2.isdigit(),
                '-2:postag': postag2,
            })
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.update({
                '-1:word[-3:]': word1[-3:],
                '-1:word[-2:]': word1[-2:],
                '-1:word[:3]': word1[:3],
                '-1:word[:2]': word1[:2],
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.islower()': word1.islower(),
                '-1:word.isupper()': word1.isupper(),
                '-1:word.isdigit()': word1.isdigit(),
                '-1:postag': postag1,
            })
        if i == 0:
            features['BOS'] = True

        if i < len(sent) - 2:
            word2 = sent[i + 2][0]
            postag2 = sent[i + 2][1]
            features.update({
                '+2:word[-3:]': word2[-3:],
                '+2:word[-2:]': word2[-2:],
                '+2:word[:3]': word2[:3],
                '+2:word[:2]': word2[:2],
                '+2:word.lower()': word2.lower(),
                '+2:word.istitle()': word2.istitle(),
                '+2:word.islower()': word2.islower(),
                '+2:word.isupper()': word2.isupper(),
                '+2:word.isdigit()': word2.isdigit(),
                '+2:postag': postag2,
            })
        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.update({
                '+1:word[-3:]': word1[-3:],
                '+1:word[-2:]': word1[-2:],
                '+1:word[:3]': word1[:3],
                '+1:word[:2]': word1[:2],
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.islower()': word1.islower(),
                '+1:word.isupper()': word1.isupper(),
                '+1:word.isdigit()': word1.isdigit(),
                '+1:postag': postag1,
            })
        if i == len(sent) - 1:
            features['EOS'] = True

        return features

    def _preprocessor(self, data):
        """ Helper function for interconversions.
        """

        tokens, labels = transform_annotated_documents_to_bio_format(data)
        pos_tags = []
        for t_i in tokens:
            pos_tags.append(tokens_to_pos_tags(t_i))
        sentences = []
        for i in range(len(tokens)):
            sentence = [(text, pos, label)
                        for text, pos, label in
                        zip(tokens[i], pos_tags[i], labels[i])]
            sentences.append(sentence)

        features = [self._sent_to_features(s) for s in sentences]
        labels = [self._sent_to_labels(s) for s in sentences]

        return features, tokens, labels

    def _sent_to_features(self, sent):
        return [self._word_to_features(sent, i) for i in range(len(sent))]

    def _sent_to_labels(self, sent):
        return [label for token, postag, label in sent]

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = os.path.join(file_path, "CRF_NER")

        model_filename = "CRF.model"
        model_save_path = os.path.join(save_path, model_filename)

        metadata_filename = "CRF_metadata.json"
        metadata_save_path = os.path.join(save_path, metadata_filename)

        mkdir(save_path)

        joblib.dump(self.crf, model_save_path)

        with open(metadata_save_path, "w") as fp:
            fp.write(json.dumps({
                "entity_label": self.entity_label
            }))

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = os.path.join(file_path, "CRF_NER")

        model_filename = "CRF.model"
        model_load_path = os.path.join(load_path, model_filename)

        metadata_filename = "CRF_metadata.json"
        metadata_load_path = os.path.join(load_path, metadata_filename)

        self.crf = joblib.load(model_load_path)

        with open(metadata_load_path, "r") as fp:
            init_metadata = json.loads(fp.read().strip())
        self.entity_label = init_metadata["entity_label"]
