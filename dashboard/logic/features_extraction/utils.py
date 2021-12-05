from math import nan


def safe_div(a, b):
    return nan if b == 0 else a / b
