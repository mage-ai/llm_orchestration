documents3 = []

for tup in documents_to_use:
    chunk = tup[3]
    tokens = model.encode_as_pieces(chunk)
    documents3.append(tup + [tokens])
