import pandas as pd


def flatten_list(_list):
    """Flatten a nested list"""
    return [x for el in _list for x in el]


def flatten_freq(_list, normalised=False):
    """Flatten a nested list and return element frequencies"""
    return pd.Series(flatten_list(_list)).value_counts(normalised)
