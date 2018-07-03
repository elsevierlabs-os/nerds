import regex as re

NON_ALPHANUMERIC_REGEXP = re.compile("[^\s\d\pL]")
MULTIPLE_WHITESPACES_REGEXP = re.compile("\s+")


def replace_non_alphanumeric(string, repl=""):
    """ Replaces all non-alphanumeric characters in a string.

        A non-alphanumeric character is anything that is not a letter in any
        language (UTF-8 characters included), or a number. The default behavior
        of this function is to eliminate them, i.e. replace them with the empty
        string (''), but the replacement string is a parameter.

        Args:
            string (str): The input string where all non-alphanumeric
                characters will be replaced.
            repl (str, optional): The replacement string for the
                non-alphanumeric characters. Defaults to ''.

        Returns:
            str: The input string, where the non-alphanumeric characters have
                been replaced.
    """
    return NON_ALPHANUMERIC_REGEXP.sub(repl, string).strip()


def eliminate_multiple_whitespaces(string, repl=" "):
    """ If a string contains multiple whitespaces, this function eliminates them.

        The default behavior of this function is to replace the multiple
        whitespaces with a single space (' '), but the replacement string is a
        parameter.

        Args:
            string (str): The input string where multiple whitespaces will be
                eliminated.
            repl (str, optional): The replacement string for multiple
                whitespaces. Defaults to ' '.

        Returns:
            str: The input string where the multiple whitespaces have been
                eliminated.
    """
    return MULTIPLE_WHITESPACES_REGEXP.sub(repl, string).strip()
