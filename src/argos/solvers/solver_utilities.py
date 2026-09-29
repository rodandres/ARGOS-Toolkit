class OdeResult(dict):
    """
    Container for ordinary differential equation integration results.

    The result behaves as a dictionary while also allowing stored values to
    be accessed as attributes.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)