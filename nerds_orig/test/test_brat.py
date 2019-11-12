import shutil

from nose.tools import assert_equal

from nerds.core.model.input.brat import BratInput
from nerds.core.model.output.brat import BratOutput


def test_brat_input():
    brat_input = BratInput("nerds/test/data/brat")
    annotated_docs = brat_input.transform()[0]

    assert_equal(
        annotated_docs.plain_text_,
        "This is a file which has an ORG and a GRANT.\n")

    assert_equal(len(annotated_docs.annotations), 2)
    assert_equal(annotated_docs.annotations[0].text, "ORG")
    assert_equal(annotated_docs.annotations[1].text, "GRANT")
    assert_equal(annotated_docs.annotations[0].label, "Organization")
    assert_equal(annotated_docs.annotations[1].label, "Grant")


def test_brat_output():
    brat_input = BratInput("nerds/test/data/brat")
    annotated_docs = brat_input.transform()

    brat_output = BratOutput("nerds/test/data/brat/tmp")
    brat_output.transform(annotated_docs)

    # Now check consistency against original file.
    brat_input_from_output = BratInput("nerds/test/data/brat/tmp")
    annotated_docs_from_output = brat_input_from_output.transform()

    assert_equal(annotated_docs[0].annotations[0].text,
                 annotated_docs_from_output[0].annotations[0].text)
    assert_equal(annotated_docs[0].annotations[0].label,
                 annotated_docs_from_output[0].annotations[0].label)
    assert_equal(annotated_docs[0].annotations[1].text,
                 annotated_docs_from_output[0].annotations[1].text)
    assert_equal(annotated_docs[0].annotations[1].label,
                 annotated_docs_from_output[0].annotations[1].label)
    assert_equal(annotated_docs[0].plain_text_,
                 annotated_docs_from_output[0].plain_text_)

    # Cleanup.
    shutil.rmtree("nerds/test/data/brat/tmp")
