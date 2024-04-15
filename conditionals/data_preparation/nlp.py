import pytextrank
import spacy

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('textrank')


@factory
def evaluate_condition(*args, **kwargs):
    return nlp
