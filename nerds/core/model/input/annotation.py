class Annotation(object):
    """ A named entity object that has been annotated on a document.

        Note:
            The attributes of this class assume a plain text representation
            of the document, after normalization. For example, `text` will
            be in lower case, if the `norm` parameter of the `to_plain_text`
            method call contains lowercasing. Similarly `offset` refers to
            the offset on the potentially normalized/clean text.

        Attributes:
            text (str): The continuous text snippet that forms the named
                entity.
            label (str): The named entity type, e.g. "PERSON", "ORGANIZATION".
            offset (2-tuple of int): Indices that represent the positions of
                the first and the last letter of the annotated entity in the
                plain text.
    """

    def __init__(self, text, label, offset):
        self.text = text
        self.label = label
        self.offset = offset

    def to_inline_string(self):
        """ Returns the annotated entity as a string of the form: label[text].
        """
        return "{}[{}]".format(self.label, self.text)

    def __str__(self):
        return "{},{} {}".format(
            self.offset[0],
            self.offset[1],
            self.to_inline_string())

    """ The following methods allow us to compare annotations for sorting,
        hashing, etc.

        Note: It doesn't make sense to compare annotations that occur in
        different documents.
    """

    def __eq__(self, other):
        return (
            (self.text == other.text) and
            (self.label == other.label) and
            (self.offset == other.offset))

    def __lt__(self, other):
        if self.offset[0] != other.offset[0]:
            return self.offset[0] < other.offset[0]
        return self.offset[1] < other.offset[1]

    def __gt__(self, other):
        if self.offset[0] != other.offset[0]:
            return self.offset[0] > other.offset[0]
        return self.offset[1] > other.offset[1]

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __hash__(self):
        return hash(
            self.text +
            self.label +
            str(self.offset[0]) +
            str(self.offset[1]))
