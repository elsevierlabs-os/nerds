from nerds.models import NERModel
from nerds.utils import get_logger, spans_to_tokens, tokens_to_spans

from spacy.util import minibatch

import itertools
import os
import random
import spacy

log = get_logger()


class SpacyNER(NERModel):

    def __init__(self,
            dropout=0.1,
            max_iter=20,
            batch_size=32):
        """ Construct a SpaCy based NER. The SpaCy library provides an EntityRecognizer 
            class to do Named Entity Recognition.

            Parameters
            ----------
            dropout : float, optional, default 0.1
                rate of dropout during training between 0 and 1.
            max_iter : int, optional, default 20
                number of epochs of training.
            batch_size : int, optional, default 32
                batch size to use during training

            Attributes
            ----------
            model_ : reference to internal SpaCy EntityRecognizer model.
        """
        super().__init__()
        self.dropout = dropout
        self.max_iter = max_iter
        self.batch_size = batch_size
        self._spacy_lm = spacy.load("en")
        self.model_ = None


    def fit(self, X, y):
        """ Trains the SpaCy NER model.

            Parameters
            ----------
            X : list(list(str))
                list of tokenized sentences, or list of list of tokens.
            y : list(list(str))
                list of list of BIO tags.

            Returns
            -------
            self
        """
        log.info("Reformatting data to SpaCy format...")
        features = [self._convert_to_spacy(tokens, labels) 
            for tokens, labels in zip(X, y)]

        log.info("Building SpaCy NER model...")
        self.model_ = spacy.blank("en")
        if "ner" not in self.model_.pipe_names:
            ner = self.model_.create_pipe("ner")
            self.model_.add_pipe(ner)
        else:
            ner = self.model_.get_pipe("ner")

        unique_labels = set()
        for _, annotations in features:
            for ent in annotations.get("entities"):
                unique_labels.add(ent[2])
                ner.add_label(ent[2])

        for label in list(unique_labels):
            ner.add_label(label)

        log.info("Training SpaCy NER model...")
        optimizer = self.model_.begin_training()

        other_pipes = [p for p in self.model_.pipe_names if p != "ner"]
        with self.model_.disable_pipes(*other_pipes):
            for it in range(self.max_iter):
                random.shuffle(features)
                losses = {}
                batches = minibatch(features, size=self.batch_size)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.model_.update(texts, annotations, 
                    sgd=optimizer, 
                    drop=self.dropout, 
                    losses=losses)
                loss_value = losses["ner"]
                log.info("Epoch: {:d} loss: {:.5f}".format(it, loss_value))

        return self


    def predict(self, X):
        """ Predicts using trained SpaCy NER model.

            Parameters
            ----------
            X : list(list(str))
                list of tokenized sentences.

            Returns
            -------
            y : list(list(str))
                list of list of predicted BIO tags.
        """
        if self.model_ is None:
            raise ValueError("Cannot predict with empty model, run fit() to train or load() pretrained model.")

        log.info("Generating predictions...")
        preds = []
        for sent_tokens in X:
            sent = " ".join(sent_tokens)
            doc = self.model_(sent)
            sent_preds = self._convert_from_spacy(sent, doc.ents)
            preds.append(sent_preds)

        return preds


    def save(self, dirpath):
        """ Save trained SpaCy NER model at dirpath.

            Parameters
            ----------
            dirpath : str
                path to model directory.

            Returns
            -------
            None
        """
        if self.model_ is None:
            raise ValueError("Cannot save empty model, run fit() to train or load() pretrained model")

        log.info("Saving model...")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.model_.to_disk(dirpath)


    def load(self, dirpath):
        """ Load a pre-trained SpaCy NER model from dirpath.

            Parameters
            ----------
            dirpath : str
                path to model directory.
            
            Returns
            -------
            self
        """
        if not os.path.exists(dirpath):
            raise ValueError("Model directory {:s} not found".format(dirpath))

        log.info("Loading model...")
        self.model_ = spacy.load(dirpath)
        return self


    def _convert_to_spacy(self, tokens, labels):
        """ Convert data and labels for single sentence to SpaCy specific format:

            Parameters
            ----------
            tokens : list(str)
                list of tokens.
            labels : list(str)
                list of BIO tags.

            Returns
            --------
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
        sentence, spans = tokens_to_spans(tokens, labels, allow_multiword_spans=False)
        return (sentence, {"entities": spans})


    def _convert_from_spacy(self, sent, entities):
        """ Converts SpaCy predictions to standard form.

            Parameters
            ----------
            sent : str
                the sentence as a string.
            entities : list(entities)
                a list of SpaCy Entity(start_char, end_char, label_) objects.

            Returns
            -------
            predictions : list(str)
                a list of BIO tags for a single sentence.
        """
        spans = [(e.start_char, e.end_char, e.label_) for e in entities]
        tokens, tags = spans_to_tokens(sent, spans, self._spacy_lm, 
            spans_are_multiword=False)
        return tags

