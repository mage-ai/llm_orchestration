import pytextrank
import spacy
from nltk.corpus import stopwords

nlp_spacy = spacy.load('en_core_web_lg')
nlp_spacy.add_pipe('textrank')

stopwords_set = set(stopwords.words('english'))


@factory
def nlp(*args, **kwargs):
    return nlp_spacy

@factory
def stop_words(*args, **kwargs) -> set:
    return stopwords_set
