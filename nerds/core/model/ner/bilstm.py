import json
import os.path

import anago
import numpy as np
from sklearn.model_selection import train_test_split

from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import AnnotatedDocument
from nerds.core.model.ner.base import NERModel
from nerds.util.convert import transform_annotated_documents_to_bio_format
from nerds.util.file import mkdir
from nerds.util.logging import get_logger
from nerds.util.nlp import sentence_to_tokens


log = get_logger()


class BidirectionalLSTM(NERModel):
    def __init__(self, entity_label=None):
        super().__init__(entity_label)
        self._label_map = {}  # Workaround for Anago splitting on dashes.
        self.key = "bilstm"

    def fit(self, X, y=None, char_emb_size=32, word_emb_size=128,
            char_lstm_units=32, word_lstm_units=128, dropout=0.1,
            batch_size=16, learning_rate=0.001, num_epochs=10):
        """ Trains the NER model. The input is a list of
            `AnnotatedDocument` instances.
        """

        # Anago splits the BIO tags on the dash "-", so if the label contains
        # a dash, it corrupts it. This is a workaround for this behavior.
        for annotated_document in X:
            for annotation in annotated_document.annotations:
                if "-" in annotation.label:
                    self._label_map[
                        annotation.label.split("-")[-1]] = annotation.label
                else:
                    self._label_map[
                        "B_" + annotation.label] = annotation.label
                    self._label_map[
                        "I_" + annotation.label] = annotation.label

        self.model = anago.Sequence(
            char_emb_size=char_emb_size, word_emb_size=word_emb_size,
            char_lstm_units=char_lstm_units, word_lstm_units=word_lstm_units,
            dropout=dropout, batch_size=batch_size,
            learning_rate=learning_rate, max_epoch=num_epochs)

        log.info("Transforming {} items to BIO format...".format(len(X)))
        training_data = transform_annotated_documents_to_bio_format(X)

        BIO_Χ = np.asarray([x_i for x_i in training_data[0] if len(x_i) > 0])
        BIO_y = np.asarray([y_i for y_i in training_data[1] if len(y_i) > 0])

        log.info("Training the BiLSTM...")
        X_train, X_valid, y_train, y_valid = train_test_split(
            BIO_Χ, BIO_y, test_size=0.1)

        self.model.train(X_train, y_train, X_valid, y_valid)
        return self

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.
        """
        annotated_documents = []
        for document in X:
            content = sentence_to_tokens(document.plain_text_)
            output = self.model.analyze(content)
            substring_index = 0
            annotations = []
            for entity in output["entities"]:
                start_idx, end_idx = _get_offsets_with_fuzzy_matching(
                    document.plain_text_, entity["text"], substring_index)
                offset = (start_idx, end_idx - 1)
                annotations.append(Annotation(
                    document.plain_text_[start_idx:end_idx],
                    self._label_map[entity["type"]],
                    offset
                ))
                substring_index = end_idx
            annotated_documents.append(AnnotatedDocument(
                document.content,
                annotations=annotations,
                encoding=document.encoding
            ))
        return annotated_documents

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = os.path.join(file_path, "BiLSTM_NER")

        model_filename = "BiLSTM.model"
        model_save_path = os.path.join(save_path, model_filename)

        metadata_filename = "BiLSTM_metadata.json"
        metadata_save_path = os.path.join(save_path, metadata_filename)

        mkdir(save_path)
        mkdir(model_save_path)

        self.model.save(model_save_path)

        with open(metadata_save_path, "w") as fp:
            fp.write(json.dumps({
                "entity_label": self.entity_label,
                "label_map": self._label_map
            }))

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = os.path.join(file_path, "BiLSTM_NER")

        model_filename = "BiLSTM.model"
        model_load_path = os.path.join(load_path, model_filename)

        metadata_filename = "BiLSTM_metadata.json"
        metadata_load_path = os.path.join(load_path, metadata_filename)

        self.model = anago.Sequence.load(model_load_path)

        with open(metadata_load_path, "r") as fp:
            init_metadata = json.loads(fp.read().strip())
        self.entity_label = init_metadata["entity_label"]
        self._label_map = init_metadata["label_map"]


def _get_offsets_with_fuzzy_matching(haystack, needle, offset_init=0):
    """ Private function for internal use in the BiLSTM implementation.

        The entities we get in the predicted text are space-separated
        tokens e.g. "anti - HIV", although in the original text the space
        may not be there e.g. "anti-HIV". If such matching occurs, this
        function will return the appropriate offsets.
    """
    search_idx = offset_init
    start_idx = None
    end_idx = None
    tokens = needle.split()

    # If this isn't a multi-term annotation, no need for fancy matchings.
    if len(tokens) == 1:
        start_idx = haystack.find(tokens[0], search_idx)
        end_idx = start_idx + len(tokens[0])
        return start_idx, end_idx

    # Otherwise we iterate the tokens 2-by-2 until we get a full match.
    token_idx = 1
    while token_idx < len(tokens):
        prv_token = tokens[token_idx - 1]
        prv_idx = haystack.find(prv_token, search_idx)
        prv_offset = prv_idx + len(prv_token)
        cur_token = tokens[token_idx]
        cur_idx = haystack.find(cur_token, prv_offset)
        # In every iteration we check if the next word starts from where
        # the previous ends.
        if cur_idx in (prv_offset, prv_offset + 1):
            if start_idx is None:
                start_idx = prv_idx
            token_idx += 1
        # If it doesn't, then it must be an accidental match, reset the index.
        else:
            start_idx = None
            token_idx = 1
        search_idx = prv_idx + len(prv_token)
    end_idx = cur_idx + len(cur_token)

    return start_idx, end_idx
