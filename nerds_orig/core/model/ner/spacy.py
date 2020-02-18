import json
import os.path
from random import shuffle

import spacy

from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import AnnotatedDocument
from nerds.core.model.ner.base import NERModel
from nerds.util.file import mkdir
from nerds.util.logging import get_logger

log = get_logger()


class SpaCyStatisticalNER(NERModel):

    def __init__(self, entity_label=None):
        super().__init__(entity_label)
        self.key = "spacy"

        # TODO: Make "en" a parameter if support for more languages is needed.
        self.nlp = spacy.blank("en")
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.create_pipe("ner")
            self.nlp.add_pipe(self.ner)

    def _transform_to_spacy_format(self, X):
        """ Transforms an annotated set of documents to the format that
            spaCy needs to operate. It's a 2-tuple of text - dictionary, where
            the dictionary has "entities" as key, and a list of tuples as
            value.

            Example:
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
        training_data = []
        for annotated_document in X:
            if len(annotated_document.annotations) == 0:
                continue
            training_record = (
                annotated_document.plain_text_,
                {"entities": []})
            # spaCy ends the offset 1 character later than we do - we consider
            # the exact index of the final character, while spaCy considers the
            # index of the cursor after the end of the token.
            for annotation in annotated_document.annotations:
                training_record[1]["entities"].append(
                    (annotation.offset[0],
                     annotation.offset[1] + 1,
                     annotation.label))
            training_data.append(training_record)
        return training_data

    def fit(self, X, y=None, num_epochs=20, dropout=0.1):
        """ Trains the NER model. The input is a list of
            `AnnotatedDocument` instances.
        """

        # In this case we're only looking for one label.
        if self.entity_label is not None:
            self.ner.add_label(self.entity_label)
        # Otherwise, add support for all the annotated labels in the set.
        else:
            label_set = set()
            for annotated_document in X:
                for annotation in annotated_document.annotations:
                    label_set.add(annotation.label)
            for unq_label in label_set:
                self.ner.add_label(unq_label)

        training_data = self._transform_to_spacy_format(X)

        # Get names of other pipes to disable them during training.
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):  # Only train NER.
            optimizer = self.nlp.begin_training()
            for _ in range(num_epochs):
                shuffle(training_data)
                losses = {}
                for text, annotations in training_data:
                    self.nlp.update([text], [annotations], sgd=optimizer,
                                    drop=dropout, losses=losses)
                log.debug("Losses: {}".format(losses))
        return self

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.
        """
        annotated_documents = []
        for document in X:
            annotated_document = self.nlp(document.plain_text_)
            annotations = []
            for named_entity in annotated_document.ents:
                annotations.append(Annotation(
                    named_entity.text,
                    named_entity.label_,
                    (named_entity.start_char, named_entity.end_char - 1)))
            annotated_documents.append(AnnotatedDocument(
                document.content,
                annotations=annotations,
                encoding=document.encoding))
        return annotated_documents

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path. """
        save_path = os.path.join(file_path, "SpaCy_NER")

        model_filename = "SpaCy.model"
        model_save_path = os.path.join(save_path, model_filename)

        metadata_filename = "SpaCy_metadata.json"
        metadata_save_path = os.path.join(save_path, metadata_filename)

        mkdir(save_path)

        self.nlp.to_disk(model_save_path)

        with open(metadata_save_path, "w") as fp:
            fp.write(json.dumps({
                "entity_label": self.entity_label
            }))

    def load(self, file_path):
        """ Loads a model saved locally. """
        load_path = os.path.join(file_path, "SpaCy_NER")

        model_filename = "SpaCy.model"
        model_load_path = os.path.join(load_path, model_filename)

        metadata_filename = "SpaCy_metadata.json"
        metadata_load_path = os.path.join(load_path, metadata_filename)

        self.nlp = spacy.load(model_load_path)

        with open(metadata_load_path, "r") as fp:
            init_metadata = json.loads(fp.read().strip())
        self.entity_label = init_metadata["entity_label"]
