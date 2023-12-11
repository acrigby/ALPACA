for name in dir():
    if not name.startswith('_'):
        del globals()[name]