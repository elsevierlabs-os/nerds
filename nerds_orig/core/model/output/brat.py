import codecs
from os.path import join

from nerds.util.file import mkdir


class BratOutput(object):
    """
    Writes data from produced annotation files to a specified folder.

    Attributes:
        path_to_folder (str): The path to the output folder where the files
            will be written.
    """

    def __init__(self, path_to_folder):
        self.path_to_folder = path_to_folder
        mkdir(self.path_to_folder)

    def transform(self, X, y=None):
        """
        Transforms the available documents into brat files.
        """
        for i, doc in enumerate(X):
            path_to_txt_file = join(self.path_to_folder, "%s.txt" % i)
            path_to_ann_file = join(self.path_to_folder, "%s.ann" % i)

            with codecs.open(path_to_txt_file, "w", doc.encoding) as f:
                f.write(doc.plain_text_)

            self._write_brat_ann_file(doc.annotations, path_to_ann_file)

        return None

    def _write_brat_ann_file(self, annotations, path_to_ann_file):
        """ Helper function to write brat annotations.
            TODO: Right now, it writes only ENTITIES to BRAT ann files,
            but we need to extend it to also write ENTITY RELATIONSHIPS.
        """

        with codecs.open(path_to_ann_file, "w", "utf-8") as f:
            for i, annotation in enumerate(annotations):

                # Must be exactly 3 things, if they are entity related.
                # e.g.: "TEmma2\tGrant 475 491\tGIA G-14-0006063".
                tag = "T%s" % i
                txt = annotation.text
                label = annotation.label
                # This is how brat's offsets are!
                start_offset = annotation.offset[0]
                end_offset = annotation.offset[1] - 1

                to_write = u"%s\t%s %s %s\t%s\n" % (tag,
                                                    label,
                                                    start_offset,
                                                    end_offset,
                                                    txt)
                f.write(to_write)
