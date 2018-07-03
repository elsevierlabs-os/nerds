
class ExactListParam(object):
    """
    An object holding valuable information about:
    1) The type of parameter values being passed.
    2) What values they are.

    In this particular case, this holds information about the fact that the
    values are a list of param options which is to be ingested as is.
    """

    def __init__(self, list_of_values):
        self.list_of_values = list_of_values


class RangeParam(object):
    """
    An object holding valuable information about:
    1) The type of parameter values being passed.
    2) What values they are.

    In this particular case, this holds information about the fact that the
    values are the high and low values of a continuous range.
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high
