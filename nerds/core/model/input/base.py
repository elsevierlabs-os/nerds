import os
from os.path import isfile, join

from sklearn.base import BaseEstimator, TransformerMixin

from nerds.core.model.input.document import Document


class DataInput(BaseEstimator, TransformerMixin):
    """ Provides the data input of a NER pipeline.

        This class provides the input to the rest of the pipeline,
        by transforming a collection of files (in the provided path) into a
        collection of documents. The `annotated` parameter differentiates
        between the already annotated input (required for training/evaluation)
        and the non-annotated input (required for entity extraction).

        Attributes:
            path_to_files (str): The path containing the input files,
                annotated or not.
            annotated (bool): If `False`, then the returned collection will
                consist of Document objects. If `True`, it will consist of
                AnnotatedDocument objects.
            encoding (str, optional): Specifies the encoding of the plain
                text. Defaults to 'utf-8'.

        Raises:
            IOError: If `path_to_files` does not exist.
    """

    def __init__(self, path_to_files, annotated=True, encoding="utf-8"):
        if not os.path.isdir(path_to_files):
            raise IOError("Invalid path for parameter 'path_to_file'")

        self.path = path_to_files
        self.annotated = annotated
        self.encoding = encoding

    def fit(self, X=None, y=None):
        # Do nothing, just return the piped object.
        return self

    def transform(self, X=None, y=None):
        """ Transforms the available documents into the appropriate objects,
            differentiating on the `annotated` parameter.
        """
        docs = []
        if self.annotated:
            raise NotImplementedError
        else:
            for found in os.listdir(self.path):
                f = join(self.path, found)
                if f.endswith(".txt") and isfile(f):
                    with open(f, 'rb') as doc_file:
                        docs.append(Document(doc_file.read(), self.encoding))
            return docs
