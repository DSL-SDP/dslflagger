"""Utility helpers for ``dslflagger.sim``."""


def _listify(val):
    """Return ``val`` as a list.

    A scalar becomes a single-element list; an existing list is returned
    unchanged. Tuples and other iterables are also wrapped as a list.
    """
    if isinstance(val, list):
        return val
    if isinstance(val, (tuple, set, frozenset)):
        return list(val)
    return [val]
