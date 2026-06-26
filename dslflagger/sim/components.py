"""Component decorator used by RFI model classes.

The :func:`component` decorator wires up the keyword-driven configuration
pattern that ``sim/rfi.py`` relies on. It:

* captures every keyword argument passed to ``__init__`` and stores it on
  ``self._kwargs``;
* adds the small ``_check_kwargs`` / ``_extract_kwarg_values`` /
  ``_listify_params`` helpers that the concrete RFI models call inside
  their ``__call__`` methods.

The decorator is intentionally minimal — it exists to make
``dslflagger.sim.rfi`` importable and to keep the keyword-driven
configuration pattern of the existing RFI classes working. A no-op
``__init__`` (e.g. the ``pass`` body of the abstract ``RFI`` base) is
detected and skipped so that double-decoration of the base class does not
break construction.
"""

import inspect


def _is_noop_init(fn):
    """Return True when ``fn`` is effectively ``object.__init__`` (or a
    ``pass`` body) — i.e. it does not accept any arguments beyond ``self``.

    For such callables we must not forward extra positional/keyword
    arguments, otherwise ``object.__init__`` raises ``TypeError``.
    """
    if fn is object.__init__:
        return True
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return False
        # Any other parameter (positional, keyword-only) means the
        # original __init__ can actually receive the forwarded args.
        return False
    return True


def component(cls):
    """Class decorator for RFI model classes.

    Replaces ``__init__`` with a wrapper that:

    1. copies the keyword arguments onto ``self._kwargs`` (so they can be
       introspected later by ``_extract_kwarg_values``);
    2. forwards ``*args`` / ``**kwargs`` to the original ``__init__`` only
       when that init is able to accept them; no-op base inits (such as
       the abstract ``RFI`` class) are skipped to avoid spurious
       ``TypeError`` from ``object.__init__``.
    """

    orig_init = cls.__init__
    noop = _is_noop_init(orig_init)

    def __init__(self, *args, **kwargs):
        self._kwargs = dict(kwargs)
        if not noop:
            orig_init(self, *args, **kwargs)

    def _check_kwargs(self, **kwargs):
        """Validate that all ``kwargs`` were registered at construction."""
        unknown = set(kwargs) - set(self._kwargs)
        if unknown:
            raise TypeError(
                "%s got unexpected keyword argument(s) %s. "
                "Valid: %s"
                % (type(self).__name__, sorted(unknown), sorted(self._kwargs))
            )

    def _extract_kwarg_values(self, **kwargs):
        """Return the values of all stored kwargs in declaration order.

        For each keyword, the value from ``kwargs`` (if provided) is used;
        otherwise the value supplied at construction is returned. This
        lets callers override individual parameters per call.
        """
        return tuple(kwargs.get(name, default) for name, default in self._kwargs.items())

    def _listify_params(self, bands, *args):
        """Broadcast per-band parameters to match ``len(bands)``.

        Delegates the scalar-to-list conversion to
        :func:`dslflagger.sim.utils._listify`.
        """
        from .utils import _listify

        listified = []
        for arg in args:
            arg = _listify(arg)
            if len(arg) == 1:
                arg = list(arg) * len(bands)
            if len(arg) != len(bands):
                raise ValueError(
                    "%s: parameter length %d does not match expected %d"
                    % (type(self).__name__, len(arg), len(bands))
                )
            listified.append(list(arg))
        return listified

    cls.__init__ = __init__
    cls._check_kwargs = _check_kwargs
    cls._extract_kwarg_values = _extract_kwarg_values
    cls._listify_params = _listify_params
    return cls
