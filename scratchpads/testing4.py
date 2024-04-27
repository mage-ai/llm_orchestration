import random

import nltk
import openai
from nltk.tokenize import TreebankWordTokenizer

from default_repo.llm_orchestration.utils.tokenization import embeddings_sum, named_entity_recognition_tokens

# tokenizer = TreebankWordTokenizer()

"""
44
Mage vs Prefect

38
is mage free if I self host?

35
for transforming and integrating data.
"""

random_idx = random.randrange(0, len(documents2))
# random_idx = len(documents2) - 1
print(random_idx)
text = documents2[random_idx][1]
query = documents2[random_idx][3]

# 84
query = """
You can limit the concurrency of the block execution to reduce resource consumption.


"""

query = query.strip()

print(query)
print('\n')

# vectors = []
# tokenized_text = tokenizer.tokenize(query)
# named_entity_recognition_tokens(nlp(query))

# tokens_list = [
#     # tokenized_text,
#     model.encode_as_pieces(query),
# ]


# print(documents2[random_idx][4])

# model = spm.SentencePieceProcessor()
# model.load('/home/src/default_repo/llm_orchestration/assets/models/notebook/subword_tokenizer.model')
# tokens2 = model.encode_as_pieces(query)
# print(tokens2)
# # print(documents2[random_idx][4][0])

# vector = openai.Embedding.create(input=tokens2, model='text-embedding-3-large').data[0].embedding

# print(documents2[random_idx][5][:20])
# print(vector[:20])

# query_embedding = np.array([vector]).astype('float64')

# print(query_embedding[0][:20])

encoded3 = tokenizer.encode_plus(query, return_tensors='pt')

# Get the subword token ids
token_ids3 = encoded3['input_ids']

# Get the attention mask (required for BERT)
attention_mask3 = encoded3['attention_mask']

# Feed the token ids and attention mask to the BERT model
with torch.no_grad():
    outputs3 = model(token_ids3, attention_mask=attention_mask3)

# Get the embeddings for each subword token
embeddings3 = outputs3.last_hidden_state

matrix3 = embeddings3[0].numpy()
vector3 = embeddings_concatenate([
    embeddings_mean(matrix3),
    embeddings_max_pooling(matrix3),
])
query_embedding = np.array([vector3])

distances, indices = index.search(query_embedding, k=5)

for distance, i in zip(distances[0], indices[0]):
    document_found = documents2[i]
    print('--------------------')
    print(distance, i, document_found[0])
    print('=========================')
    print(document_found[3])
    # print(document_found[5][:20])
    print('--------------------')
    print('\n')

# from default_repo.llm_orchestration.utils.chunking import chunk_sentences


# # print(
# #     len(chunk_sentences(nlp(text))),
# #     len(nltk.sent_tokenize(text)),
# # )

# for c in chunk_sentences(nlp(text)):
#     print('\n---------------------------------------\n')
#     print(c)

# print('\n=================================\n')

# for c in nltk.sent_tokenize(text):
#     print('\n---------------------------------------\n')
#     print(c)

#  == v3[:20]
# documents2[random_idx][5][:] == documents2[21][5][:], documents2[random_idx][4][:] == documents2[21][4][:]

# for doc in documents2:
#     v1_ = documents2[random_idx][5]
#     v2_ = doc[5]
#     print(v1 == v2, sum(v1), sum(v2))