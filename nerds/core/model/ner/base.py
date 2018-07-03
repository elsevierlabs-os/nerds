from sklearn.base import BaseEstimator, ClassifierMixin

from nerds.core.model.input.document import AnnotatedDocument


class NERModel(BaseEstimator, ClassifierMixin):
    """ Provides a basic interface to train NER models and annotate documents.

        This is the core class responsible for training models that perform
        named entity recognition, and retrieving named entities from documents.
    """

    def __init__(self, entity_label=None):
        self.entity_label = entity_label
        self.key = ""  # To be added in subclass.

    def fit(self, X, y=None):
        """ Trains the NER model. The input is a list of
            `AnnotatedDocument` instances.

            The basic implementation of this method performs no training and
            should be overridden by offspring.
        """
        return self

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.

            The basic implementation of this method does not annotate any
            entities and should be overridden by offspring.
        """
        annotated_documents = []
        for document in X:
            annotated_documents.append(AnnotatedDocument(
                document.content,
                encoding=document.encoding))
        return annotated_documents

    def extract(self, X, y=None):
        """ Returns a list of entities, extracted from annotated documents. """
        annotated_documents = self.transform(X, y)
        entities = []
        for annotated_document in annotated_documents:
            entities.append(annotated_document.annotations)
        return entities

    def save(self, file_path):
        """ Saves a model to the local disk, provided a file path.
            Should be overridden.
        """
        raise NotImplementedError

    def load(self, file_path):
        """ Loads a model saved locally. Should be overridden. """
        raise NotImplementedError
