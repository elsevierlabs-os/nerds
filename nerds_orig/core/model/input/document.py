class Document(object):
    """ Represents a basic input document in the extraction pipeline.

        This is an abstraction that is meant to be extended in order to support
        a variety of document types. Offspring of this class should implement
        the method `to_plain_text` that transforms the input document into its
        plain text representation. The present class returns the text itself,
        thus it can be used with any simple `.txt` file.

        Attributes:
            content (bytes): The byte representation of the document object as
                read from the input stream (e.g. with the `rb` flag).
            encoding (str, optional): Specifies the encoding of the plain
                text. Defaults to 'utf-8'.

        Raises:
            TypeError: If `content` is not a byte stream.
    """

    def __init__(self, content, encoding='utf-8'):
        if isinstance(content, bytes):
            self.content = content
        else:
            raise TypeError("Invalid type for parameter 'content'.")
        self.encoding = encoding

    @property
    def plain_text_(self):
        """ Method that transforms a document into its plain text representation.

            Returns:
                str: The content of this document as plain text.
        """
        return self.content.decode(self.encoding)


class AnnotatedDocument(Document):
    """ Represents a document object that has been annotated with entities.

        This class serves primarily two purposes: (i) it holds the training
        data for the NER model, and (ii) it represents an input (unobserved)
        document __after__ it has been annotated by the pipeline.

        Attributes:
            content (bytes): The byte representation of the document object as
                read from the input stream (e.g. with the `rb` flag).
            annotations (list(Annotation), optional): The annotations on the
                plain text representation of the document. If None, it defaults
                to an empty list.
    """

    def __init__(self, content, annotations=None, encoding='utf-8'):
        super().__init__(content, encoding)
        self.annotations = annotations or []

    @property
    def annotated_text_(self):
        """ Method that returns the document's text with inline annotated entities.

            Yields:
                str: Every line of text in the input document, where the
                    entities have been replaced with inline annotations.
        """
        cur_annotation_idx = 0
        text_idx = 0
        annotated_line = ""
        while cur_annotation_idx < len(self.annotations):
            # Iteratively append chunks of text plus the annotation.
            cur_annotation = self.annotations[cur_annotation_idx]
            annotated_line += (
                self.plain_text_[text_idx:cur_annotation.offset[0]] +
                cur_annotation.to_inline_string())
            text_idx = cur_annotation.offset[1] + 1
            cur_annotation_idx += 1
        else:
            # If no annotations are left, append the rest of the text.
            annotated_line += self.plain_text_[text_idx:]
        return annotated_line
