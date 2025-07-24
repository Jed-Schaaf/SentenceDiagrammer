"""Loads Natural Language Processor to parse sentences"""
import spacy
import benepar

# Load spaCy with benepar
nlp = spacy.load("en_core_web_sm")
try:
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
except ValueError:
    benepar.download("benepar_en3")
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

def split_sentences(sentence):
    """
    Parse the input sentence and return the spaCy Doc object containing its grammatical structure.
    """
    doc = nlp(sentence)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def parse_sentence(sentence):
    """
    Parse the input sentence and return the spaCy Doc object containing its grammatical structure.
    """
    doc = nlp(sentence)
    return doc
