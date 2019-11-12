import os
from os.path import isfile, join

from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.base import DataInput
from nerds.core.model.input.document import AnnotatedDocument


class BratInput(DataInput):
    """
    Reads input data from a collection of BRAT txt/ann files

    This class provides the input to the rest of the pipeline,
    by transforming a collection of BRAT txt/ann files (in the provided path)
    into collection of documents. The `annotated` parameter differentiates
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
    """

    def __init__(self, path_to_files, annotated=True, encoding="utf-8"):
        super().__init__(path_to_files, annotated, encoding)

    def transform(self, X=None, y=None):
        """ Transforms the available documents into the appropriate objects,
            differentiating on the `annotated` parameter.
        """

        # If not annotated, fall back to base class and simply read files
        if not self.annotated:
            return super().transform(X, y)
        # Else, read txt/ann
        else:
            annotated_docs = []
            for found in os.listdir(self.path):
                f = join(self.path, found)

                if f.endswith(".txt") and isfile(f):
                    # Standard brat folder structure:
                    # For every txt there should be an ann.
                    brat_f = f.replace(".txt", ".ann")
                    annotations = self._read_brat_ann_file(brat_f)

                    with open(f, 'rb') as doc_file:
                        annotated_docs.append(AnnotatedDocument(
                            doc_file.read(),
                            annotations,
                            self.encoding))

            return annotated_docs

    def _read_brat_ann_file(self, path_to_ann_file):
        """ Helper function to read brat annotations.
            TODO: Right now, it reads only ENTITIES from BRAT ann files,
            but we need to extend it to also read ENTITY RELATIONSHIPS.
        """

        annotations = set()

        if isfile(path_to_ann_file):
            with open(path_to_ann_file, 'rb') as ann_file:
                for ann_line in ann_file:
                    ann_line = ann_line.decode(self.encoding)
                    # Comments start with a hash
                    if ann_line.startswith("#"):
                        continue

                    split_ann_line = ann_line.strip().split("\t")

                    # Must be exactly 3 things, if they are entity related.
                    # e.g.: "TEmma2\tGrant 475 491\tGIA G-14-0006063"
                    # The annotations can also be relations.
                    # TODO: Add code to read relations.
                    if len(split_ann_line) > 2:
                        entity_str = split_ann_line[2]

                        # Looks like "Grant 475 491"
                        entity_type_offsets = split_ann_line[1].split(" ")
                        entity_name = entity_type_offsets[0]
                        start_offset = int(entity_type_offsets[1])
                        end_offset = int(entity_type_offsets[2]) - 1

                        annotations.add(Annotation(
                            entity_str, entity_name,
                            (start_offset, end_offset)))

        return sorted(list(annotations))
