import pytextrank
import spacy
from transformers import BertTokenizer
from nltk.corpus import stopwords

nlp_spacy = spacy.load('en_core_web_lg')
nlp_spacy.add_pipe('textrank')

stopwords_set = set(stopwords.words('english'))

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


@factory
def nlp(*args, **kwargs):
    return nlp_spacy


@factory
def stop_words(*args, **kwargs) -> set:
    return stopwords_set


@factory
def tokenizer(*args, **kwargs):
    return bert_tokenizer