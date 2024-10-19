
def empty():
    raise StopIteration
    yield


x = empty()

next(x)