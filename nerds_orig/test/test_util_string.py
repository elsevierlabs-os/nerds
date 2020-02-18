from nose.tools import assert_equal

from nerds.util.string import (
    replace_non_alphanumeric, eliminate_multiple_whitespaces)


def test_replace_non_alphanumeric():
    non_alphanumeric = "this is 1 invalid character ^"
    # U+0FC9 is a Tibetan symbol - should be removed
    non_alphanumeric_unicode = "this is 1 invalid character \u0fc9"
    # U+0298 is a Latin letter - should be kept
    alphanumeric_unicode = "this is 1 valid character \u0298"

    result = replace_non_alphanumeric(non_alphanumeric)
    assert_equal(result, "this is 1 invalid character")

    result = replace_non_alphanumeric(non_alphanumeric_unicode)
    assert_equal(result, "this is 1 invalid character")

    result = replace_non_alphanumeric(alphanumeric_unicode)
    assert_equal(result, "this is 1 valid character \u0298")


def test_eliminate_multiple_whitespaces():
    four_spaces = "space    space"
    one_newline = "space\nspace"
    four_newlines = "space\n\n\n\nspace"
    one_tab = "space\tspace"
    four_tabs = "space\t\t\t\tspace"
    mixed_whsp = "space\t\t  \n\nspace"

    result = eliminate_multiple_whitespaces(four_spaces)
    assert_equal(result, "space space")

    result = eliminate_multiple_whitespaces(one_newline)
    assert_equal(result, "space space")

    result = eliminate_multiple_whitespaces(four_newlines)
    assert_equal(result, "space space")

    result = eliminate_multiple_whitespaces(one_tab)
    assert_equal(result, "space space")

    result = eliminate_multiple_whitespaces(four_tabs)
    assert_equal(result, "space space")

    result = eliminate_multiple_whitespaces(mixed_whsp)
    assert_equal(result, "space space")
