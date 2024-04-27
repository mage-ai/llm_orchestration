@transformer
def transform(data, *args, **kwargs):
    arr = []
    for d in data:
        arr += d

    return [
        arr,
    ]