"""From gymnasium"""
"""Class for pickling and unpickling objects via their constructor arguments."""


class EzPickle:


    def __init__(self, *args, **kwargs):
        """Uses the ``args`` and ``kwargs`` from the object's constructor for pickling."""
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        """Returns the object pickle state with args and kwargs."""
        return {
            "_ezpickle_args": self._ezpickle_args,
            "_ezpickle_kwargs": self._ezpickle_kwargs,
        }

    def __setstate__(self, d):
        """Sets the object pickle state using d."""
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)