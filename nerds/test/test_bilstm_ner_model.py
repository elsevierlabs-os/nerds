from nose.tools import assert_equal

from nerds.core.model.ner.bilstm import _get_offsets_with_fuzzy_matching


def test_get_offsets_with_fuzzy_matching():
    # Repetition of "quick" and "fox" on purpose.
    test_str = "The quick quick-brown fox jumps over the lazy fox."
    look_for = "quick - brown fox"
    expected = (10, 25)
    result = _get_offsets_with_fuzzy_matching(test_str, look_for)
    assert_equal(result, expected)
