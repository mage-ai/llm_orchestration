from default_repo.llm_orchestration.models.subword import get_train_transform


documents_to_use = documents[:100]
print(len(documents_to_use))
get_train_transform(nlp, [d[1] for d in documents_to_use], train=True, vocab_size=300)['model']
model = spm.SentencePieceProcessor()
model.load('/home/src/default_repo/llm_orchestration/assets/models/notebook/subword_tokenizer.model')