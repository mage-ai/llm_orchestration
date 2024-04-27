from default_repo.llm_orchestration.utils.tokenization import embeddings_concatenate, embeddings_max_pooling, embeddings_mean


# len(embeddings[0][1])


# import openai

# from default_repo.llm_orchestration.utils.tokenization import embeddings_sum

# # import sentencepiece as spm

# # model = spm.SentencePieceProcessor()
# # model.load('/home/src/default_repo/llm_orchestration/assets/models/notebook/subword_tokenizer.model')


documents2 = []

for idx, data in enumerate(documents_to_use):
    document_id, document, metadata, chunk, _ = data

    # print(len(tokens))

    # tokens = model.encode_as_pieces(chunk)
    encoded = tokenizer.encode_plus(chunk, return_tensors='pt')

    # Get the subword token ids
    token_ids = encoded['input_ids']

    # Get the attention mask (required for BERT)
    attention_mask = encoded['attention_mask']

    # Feed the token ids and attention mask to the BERT model
    with torch.no_grad():
        outputs = model(token_ids, attention_mask=attention_mask)

    # Get the embeddings for each subword token
    embeddings = outputs.last_hidden_state

    matrix = embeddings[0].numpy()
    vector = embeddings_concatenate([
        embeddings_mean(matrix),
        embeddings_max_pooling(matrix),
    ])

    
    
    # vector = openai.Embedding.create(input=tokens, model='text-embedding-3-large').data[0].embedding
    # print(sum(vector))

    documents2.append([
        document_id, 
        document, 
        metadata, 
        chunk, 
        encoded, 
        vector,
    ])

    if idx % 5 == 0:
        print(len(documents2), len(documents_to_use))

# embedding = []
# for token in documents3[0][4]:
#     if token in model.wv:
#         embedding.append(model.wv[token])
#     else:
#         # If the token is not in the vocabulary, you can either ignore it or use a special "unknown" token vector
#         embedding.append(model.wv["unk"])
attention_mask