import csv

from nerds.core.model.input.document import Document
from nerds.util.convert import transform_bio_tags_to_annotated_document


PATH_TO_FILE = "train.csv"


def read_kaggle_data():
    sentences = []
    pos = []
    tag = []

    tmp_sentence = []
    tmp_pos = []
    tmp_tag = []

    with open(PATH_TO_FILE, "rt") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # Ignore the header
        next(csv_reader)

        for row in csv_reader:

            if row[0].startswith("Sentence: "):
                if len(tmp_sentence) != 0:
                    sentences.append(tmp_sentence)
                    pos.append(tmp_pos)
                    tag.append(tmp_tag)

                tmp_sentence = []
                tmp_pos = []
                tmp_tag = []

            tmp_sentence.append(row[1])
            tmp_pos.append(row[2])
            tmp_tag.append(row[3].replace("-", "_"))

        if len(tmp_sentence) != 0:
            sentences.append(tmp_sentence)
            pos.append(tmp_pos)

    return sentences, pos, tag


def data_to_annotated_docs():
    sentences, pos, tags = read_kaggle_data()

    documents = [Document(u" ".join(sentence).encode("utf-8"))
                 for sentence in sentences]

    ann_docs = []
    for i in range(len(documents)):
        try:
            sentence = sentences[i]
            tag = tags[i]
            document = documents[i]
            ann_docs.append(transform_bio_tags_to_annotated_document(sentence,
                                                                     tag,
                                                                     document))
        except IndexError:
            continue
    return ann_docs
