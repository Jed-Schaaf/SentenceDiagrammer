import spacy

# Load spaCy with benepar
nlp = spacy.load("en_core_web_sm")
try:
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
except Exception:
    print("Please install benepar and download the model: 'import benepar; benepar.download(\"benepar_en3\")'")

def parse_sentence(sentence):
    """
    Parse the input sentence and return the spaCy Doc object containing its grammatical structure.
    """
    doc = nlp(sentence)
    return doc