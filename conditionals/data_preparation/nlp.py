# import pytextrank
import spacy
# from transformers import BertTokenizer
# from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_lg')
# nlp.add_pipe('textrank')

# stopwords_set = set(stopwords.words('english'))

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


@factory
def nlp(*args, **kwargs):
    return nlp


@factory
def stop_words(*args, **kwargs) -> set:
    pass
    # return stopwords_set


@factory
def tokenizer(*args, **kwargs):
    pass
    # return bert_tokenizer
