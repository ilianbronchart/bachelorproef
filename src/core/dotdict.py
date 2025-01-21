class DotDict(dict):
    """DotDict class allows accessing dictionary keys as attributes."""

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
