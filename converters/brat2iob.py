import argparse
import operator
import os
import re
import shutil
import spacy
import tempfile

from nerds.utils import spans_to_tokens, get_logger

def segment_text_to_sentences(text_file, sentence_splitter):
    """ Segment text into sentences. Text is provided by BRAT in .txt 
        file.

        Args:
            text_file (str): the full path to the BRAT .txt file.
            sentence_splitter (spacy LM): SpaCy EN language model.

        Returns:
            sentences (list((int, int, str))): list of sentence spans. 
                Spans are triples of (start_offset, end_offset, text),
                where offset is relative to the text.
    """
    sentences = []
    ftext = open(text_file, "r")
    for line in ftext:
        splits = sentence_splitter(line.strip())
        for sent in splits.sents:
            sentences.append((sent.start_char, sent.end_char, sent.text))
    ftext.close()
    return sentences


def parse_text_annotations(ann_file):
    """ Parses BRAT annotations provided in the .ann file and converts them
        to annotation spans of (start_position, end_position, entity_class).

        Args:
            ann_file (str): full path to the BRAT .ann file.

        Returns:
            annotations (list((int, int, str))): list of annotation spans.
                Spans are triples of (start_offset, end_offset, entity_class)
                where offset is relative to the text.
    """
    annots = []
    fann = open(ann_file, "r")
    for line in fann:
        cols = re.split(r"\s+", line.strip())
        if not cols[0].startswith("T"):
            continue
        annots.append((int(cols[2]), int(cols[3]), cols[1]))
    fann.close()
    return annots


def apply_annotations(sentences, annotations, tokenizer):
    """ Apply annotation spans to the sentence spans to create a list of tokens
        and tags.

        Args:
            sentences (list((int, int, str))): list of sentence spans.
            annotations (list((int, int, str))): list of annotation spans.
            tokenizer (spacy LM): SpaCy EN language model.

        Returns:
            tokens_tags_list (list((list(str), list(str)))): list of list of token
                tag pairs. Each list of token-tag pairs corresponds to a single 
                sentence.
    """
    tokens_tags_list = []
    for sent_start, sent_end, sent_text in sentences:
        sent_annots = [a for a in annotations if a[0] >= sent_start and a[1] <= sent_end]
        # convert document offsets to sentence offsets
        sent_annots = [(s[0] - sent_start, s[1] - sent_start, s[2]) for s in sent_annots]
        tokens, tags = spans_to_tokens(sent_text, sent_annots, tokenizer)
        tokens_tags_list.append(zip(tokens, tags))
    return tokens_tags_list


def convert_brat_to_iob(input_dir, output_file, nlp):
    """ Convenience Convertor function.

        Args:
            input_dir (str): the directory where the BRAT .txt and .ann files
                are located.
            output_file (str): the full path name of file to write output in
                IOB format to.
            nlp (SpaCy LM): reference to the SpaCy EN model.

        Returns: 
            None.
    """
    fout = open(output_file, "w")
    for text_file in os.listdir(input_dir):
        # only process .txt and .ann pairs in specified directory
        if not text_file.endswith(".txt"):
            continue
        annot_file = text_file[:-4] + ".ann"
        if not os.path.exists(os.path.join(input_dir, annot_file)):
            # do not process file if no corresponding .ann file
            continue
        # process file pair
        logger.info("Processing file: {:s}".format(text_file))
        sentences = segment_text_to_sentences(os.path.join(input_dir, text_file), nlp)
        annotations = parse_text_annotations(os.path.join(input_dir, annot_file))
        tokens_tags_list = apply_annotations(sentences, annotations, nlp)
        for tokens_tags in tokens_tags_list:
            for token, tag in tokens_tags:
                fout.write("{:s}\t{:s}\n".format(token, tag))
            fout.write("\n")

    fout.close()


def do_self_test(nlp):
    """ Simple self-test with small dataset to prove that this works okay. """
    text = "Pierre Vinken, 61 years old, will join the board as a nonexecutive director, Nov. 29. Mr. Vinken is chairman of Elsevier N.V., the Dutch publishing group."
    annotations = [
        "T1	PER 0 13	Pierre Vinken",
        "T2	PER 86 96	Mr. Vinken",
        "T3	DATE 15 27	61 years old",
        "T4	DATE 77 84	Nov. 29",
        "T5	ORG 112 125	Elsevier N.V.",
        "T6	NORP 131 136	Dutch"
    ]
    input_dir = tempfile.mkdtemp(dir="/tmp")
    ftext = open(os.path.join(input_dir, "test.txt"), "w")
    ftext.write(text)
    ftext.close()
    fann = open(os.path.join(input_dir, "test.ann"), "w")
    for line in annotations:
        fann.write(line + "\n")
    fann.close()
    output_file = os.path.join(input_dir, "test.iob")
    convert_brat_to_iob(input_dir, output_file, nlp)
    fout = open(output_file, "r")
    for line in fout:
        logger.warn(line.strip())
    shutil.rmtree(input_dir)


################################ main ################################
#
# usage: brat2iob.py [-h] [-i INPUT_DIR] [-o OUTPUT_FILE] [-t]
# Script to convert BRAT annotations to IOB (NERDS) format.
# optional arguments:
#   -h, --help            show this help message and exit
#   -i INPUT_DIR, --input_dir INPUT_DIR
#                         Directory to store BRAT .txt and .ann files.
#   -o OUTPUT_FILE, --output_file OUTPUT_FILE
#                         Output file to write IOB output to.
#   -t, --test            Runs self test.
######################################################################

parser = argparse.ArgumentParser(
    description="Script to convert BRAT annotations to IOB (NERDS) format.")
parser.add_argument("-i", "--input_dir", help="Directory to store BRAT .txt and .ann files.")
parser.add_argument("-o", "--output_file", help="Output file to write IOB output to.")
parser.add_argument("-t", "--test", help="Runs self test.", action="store_true")
args = parser.parse_args()

logger = get_logger()

input_dir = args.input_dir
output_file = args.output_file
self_test = args.test

nlp = spacy.load("en")

if self_test:
    logger.info("Executing self test...")
    do_self_test(nlp)
else:
    logger.info("Reading BRAT .txt and .ann files from: {:s}".format(input_dir))
    logger.info("Writing IOB tokens/tags to file: {:s}".format(output_file))
    convert_brat_to_iob(input_dir, output_file, nlp)

