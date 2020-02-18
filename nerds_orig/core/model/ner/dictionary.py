from os.path import isfile

import ahocorasick

from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import AnnotatedDocument
from nerds.core.model.ner.base import NERModel
from nerds.util.logging import get_logger

log = get_logger()


class ExactMatchDictionaryNER(NERModel):

    def __init__(self, path_to_dictionary_file, entity_label):
        super().__init__(entity_label)
        self.key = "em-dict"

        if path_to_dictionary_file is not None:
            self.path_to_dictionary_file = path_to_dictionary_file
            self._create_automaton()
        else:
            # Must get a dictionary as an input!
            raise Exception("No dictionary provided!")

    def _create_automaton(self):

        if not isfile(self.path_to_dictionary_file):
            raise Exception("%s is not a file." % self.path_to_dictionary_file)

        # Initialize automaton.
        self.automaton = ahocorasick.Automaton()

        # Index counter.
        count = 0

        # Dictionary must be one word per line.
        log.debug("Started loading dictionary at {}".format(
            self.path_to_dictionary_file))
        with open(self.path_to_dictionary_file, 'r') as dict_file:
            for search_expr in dict_file:
                search_expr = search_expr.strip()
                if search_expr != "":
                    self.automaton.add_word(search_expr, (count, search_expr))
                    count += 1
        log.debug("Successfully loaded dictionary")

        self.automaton.make_automaton()

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.

            In a dictionary based approach, a dictionary of keywords is used
            to create a FSA which is then used to search with. See [1].
            [1]: https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm
        """
        annotated_documents = []
        for document in X:
            annotations = []
            doc_content_str = document.plain_text_
            for item in self.automaton.iter(doc_content_str):
                end_position, (index, word) = item

                start_position = (end_position - len(word) + 1)
                end_position = end_position + 1

                annotations.append(Annotation(
                    word,
                    self.entity_label,
                    (start_position, end_position)))

            annotated_documents.append(AnnotatedDocument(
                document.content,
                annotations=annotations,
                encoding=document.encoding))

        return annotated_documents


class ExactMatchMultiClassDictionaryNER(NERModel):

    def __init__(self, path_to_dictionary_file):
        super().__init__()
        self.key = "em-dict"

        if path_to_dictionary_file is not None:
            self.path_to_dictionary_file = path_to_dictionary_file
            self._create_automaton()
        else:
            # Must get a dictionary as an input!
            log.warning("No path to dictionary provided, fit() model to load")

    def _create_automaton(self):

        if not isfile(self.path_to_dictionary_file):
            raise Exception("%s is not a file." % self.path_to_dictionary_file)

        # Initialize automaton.
        self.automaton = ahocorasick.Automaton()

        # Dictionary must be one word per line.
        log.debug("Started loading dictionary at {}".format(
            self.path_to_dictionary_file))
        with open(self.path_to_dictionary_file, 'r') as dict_file:
            for line in dict_file:
                search_expr, entity_type = line.strip().split('\t')
                if search_expr != "":
                    self.automaton.add_word(search_expr, (entity_type, search_expr))
        log.debug("Successfully loaded dictionary")

        self.automaton.make_automaton()

    def fit(self, X, y=None):
        # Initialize automaton.
        self.automaton = ahocorasick.Automaton()

        # populate automaton from annotation values provided
        for annotated_document in X:
            for annotation in annotated_document.annotations:
                search_expr = annotation.text
                entity_type = annotation.label
                if search_expr != "":
                    self.automaton.add_word(search_expr, (entity_type, search_expr))
        log.debug("Successfully loaded dictionary")

        self.automaton.make_automaton()

    def transform(self, X, y=None):
        """ Annotates the list of `Document` objects that are provided as
            input and returns a list of `AnnotatedDocument` objects.

            In a dictionary based approach, a dictionary of keywords is used
            to create a FSA which is then used to search with. See [1].
            [1]: https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm
        """
        annotated_documents = []
        for document in X:
            annotations = []
            doc_content_str = document.plain_text_
            for item in self.automaton.iter(doc_content_str):
                end_position, (label, word) = item

                start_position = (end_position - len(word) + 1)
                end_position = end_position + 1

                # Aho-Corasick matches partial strings in the input document, which
                # leads to spurious matches, so we check to see that the match spans
                # a full word before adding it to our list of valid annotations
                if ((start_position <= 0 and doc_content_str[end_position] == " ") or 
                        (end_position >= len(doc_content_str) and doc_content_str[start_position - 1] == " ") or 
                        (doc_content_str[start_position - 1] == " " and doc_content_str[end_position] == " ")):
                    annotations.append(Annotation(
                        word,
                        label,
                        (start_position, end_position)))

            annotated_documents.append(AnnotatedDocument(
                document.content,
                annotations=annotations,
                encoding=document.encoding))

        return annotated_documents


