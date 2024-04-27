import faiss
import numpy as np


vectors1 = [d[5] for d in documents2]
print(len(vectors))

# print(documents2[0][5])
vectors2 = np.vstack(vectors1)   # Stack all vectors into a NumPy array
dim = vectors2.shape[1]         # Number of dimensions of each vector
# print(dim)
# print(vectors[0])
# print(documents2[0][4])
index = faiss.IndexFlatL2(dim)                 # Use the L2 distance for the quantizer

if False:
    nlist = min(100, len(vectors)) # This means that the index will partition the vector space into 100 distinct cells.
    index = faiss.IndexIVFFlat(index, dim, nlist)  # Initialize the IVF index
    index.train(vectors)  # Train the index

index.add(vectors2)  # Add vectors to the index

# v3 = openai.Embedding.create(input=documents2[-1][4], model='text-embedding-3-large').data[0].embedding

# vector[:20] == documents2[-1][5][:20]
# vector[:20] == v3[:20]

# v = vectors[0]
# for v2 in vectors:
#     print(all(v == v2))