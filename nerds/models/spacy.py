from nerds.models import NERModel
from nerds.utils import get_logger

from spacy.util import minibatch

import itertools
import os
import random
import spacy

log = get_logger()


class SpacyNER(NERModel):

    def __init__(self, entity_label=None):
        """ Build a SpaCy EntityRecognizer NER model.

            Args:
                entity_label (str, default None): entity label for single class NER.
        """
        super().__init__(entity_label)
        self.key = "spacy_ner"
        self.model = None


    def fit(self, X, y,
            num_epochs=20,
            dropout=0.1,
            batch_size=32):
        """ Trains the SpaCy NER model.

            Args:
                X (list(list(str))): list of tokenized sentences, or list of list
                    of tokens.
                y (list(list(str))): list of list of IOB tags.
                num_epochs (int): number of epochs of training.
                dropout (float): rate of dropout during training between 0 and 1.
                batch_size (int): batch size to use during training
        """
        log.info("Reformatting data to SpaCy format...")
        features = [self._convert_to_spacy(tokens, labels) 
            for tokens, labels in zip(X, y)]

        log.info("Building SpaCy NER model...")
        self.model = spacy.blank("en")
        if "ner" not in self.model.pipe_names:
            ner = self.model.create_pipe("ner")
            self.model.add_pipe(ner)
        else:
            ner = self.model.get_pipe("ner")

        unique_labels = set()
        for _, annotations in features:
            for ent in annotations.get("entities"):
                unique_labels.add(ent[2])
                ner.add_label(ent[2])

        for label in list(unique_labels):
            ner.add_label("B-" + label)
            ner.add_label("I-" + label)
        ner.add_label("O")

        log.info("Training SpaCy NER model...")
        optimizer = self.model.begin_training()

        other_pipes = [p for p in self.model.pipe_names if p != "ner"]
        with self.model.disable_pipes(*other_pipes):
            for it in range(num_epochs):
                random.shuffle(features)
                losses = {}
                batches = minibatch(features, size=batch_size)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.model.update(texts, annotations, 
                    sgd=optimizer, 
                    drop=dropout, 
                    losses=losses)
                loss_value = losses["ner"]
                log.info("Epoch: {:d} loss: {:.5f}".format(it, loss_value))

        return self


    def predict(self, X):
        """ Predicts using trained SpaCy NER model.

            Args:
                X (list(list(str))): list of tokenized sentences.
                is_featurized (bool, default False): if True, X is a list
                    of list of features, else X is a list of list of tokens.
            Returns:
                y (list(list(str))): list of list of predicted BIO tags.
        """
        if self.model is None:
            raise ValueError("Cannot predict with empty model, run fit() to train or load() pretrained model.")

        log.info("Generating predictions...")
        preds = []
        for sent_tokens in X:
            sent = " ".join(sent_tokens)
            doc = self.model(sent)
            entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
            sent_preds = self._convert_from_spacy(sent, entities)
            preds.append(sent_preds)

        return preds


    def save(self, dirpath):
        """ Save trained SpaCy NER model at dirpath.

            Args:
                dirpath (str): path to model directory.
        """
        if self.model is None:
            raise ValueError("Cannot save empty model, run fit() to train or load() pretrained model")

        log.info("Saving model...")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.model.to_disk(dirpath)


    def load(self, dirpath):
        """ Load a pre-trained SpaCy NER model from dirpath.

            Args:
                dirpath (str): path to model directory.
            Returns:
                this object populated with pre-trained model.
        """
        if not os.path.exists(dirpath):
            raise ValueError("Model directory {:s} not found".format(dirpath))

        log.info("Loading model...")
        self.model = spacy.load(MODEL_DIR)
        return self


    def _convert_to_spacy(self, tokens, labels):
        """ Convert data and labels for single sentence to SpaCy specific format:

            Args:
                tokens (list(str)): list of tokens.
                labels (list(str)): list of IOB tags.

            Returns:
                list of tuples in SpaCy format as shown below:
                (
                    "The quick brown fox jumps over the lazy dog",
                    {
                        "entities": [
                            (16, 19, "ANIMAL"),
                            (40, 43, "ANIMAL")
                        ]
                    }
                )
        """
        content = " ".join(tokens)
        offsets, current_offset = [], 0
        for token, label in zip(tokens, labels):
            start_offset = current_offset
            end_offset = start_offset + len(token)
            if label != "O":
                offsets.append((start_offset, end_offset, label.split("-")[-1]))
            current_offset = end_offset + 1 # skip space
        return (content, {"entities": offsets})


    def _entities2dict(self, entities):
        """ Convert entities returned from SpaCy into a dictionary keyed by
            offset pair. This allows predicted entities to be looked up quickly
            and the appropriate tag populated in _convert_from_spacy().
            
            Args:
                entities (list((begin, end, label))): list of label predictions.

            Return:
                entity_dict (dict{(begin, end): label})
        """
        entity_dict = {}
        for begin, end, label in entities:
            key = ":".join([str(begin), str(end)])
            entity_dict[key] = label
        return entity_dict


    def _tokenize_with_offsets(self, sent):
        """ Tokenize a sentence by space, and write out tuples that include
            offsets for each token.

            Args:
                sent (str): sentence as a string.
            
            Returns:
                offsets (list((start, end), token)): list of tokens with 
                    offset information.
        """
        tokens = sent.split()
        offsets, curr_offset = [], 0
        for token in tokens:
            begin = curr_offset
            end = begin + len(token)
            key = ":".join([str(begin), str(end)])
            offsets.append((key, token))
            curr_offset = end + 1
        return offsets


    def _convert_from_spacy(self, sent, entities):
        """ Converts SpaCy predictions to standard form.

            Args:
                sent (str): the sentence as a string.
                entities (list(entities)): a list of SpaCy Entity objects
                Entity(start_char, end_char, label_).

            Returns:
                predictions (list(str)): a list of IOB tags for a single
                    sentence.
        """
        iob_tags, prev_tag = [], None
        entity_dict = self._entities2dict(entities)
        for offset_token in self._tokenize_with_offsets(sent):
            offset, token = offset_token
            if offset in entity_dict:
                curr_tag = entity_dict[offset]
                if prev_tag is None or prev_tag != curr_tag:
                    iob_tags.append("B-" + curr_tag)
                else:
                    iob_tags.append("I-" + curr_tag)
            else:
                curr_tag = "O"
                iob_tags.append("O")
            prev_tag = curr_tag
        return iob_tags

