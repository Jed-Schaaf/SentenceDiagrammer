from graphviz import Digraph
from nltk import Tree
from nlp_module import parse_sentence

def generate_diagram(doc, style='dependency'):
    """
    Generate an SVG diagram from the parsed sentence structure.
    Supports 'dependency' and 'reed-kellogg' styles.
    """

    doc = parse_sentence(sentence)
    sent = list(doc.sents)[0]
    parse_tree = Tree.fromstring(sent._.parse_string)
    dot = Digraph()

    if style == 'dependency':
        for token in doc:
            dot.node(str(token.i), token.text)
            if token.head != token:
                dot.edge(str(token.head.i), str(token.i), label=token.dep_)

    elif style == 'reed-kellogg':
        dot.attr(rankdir='TB')  # Top-to-bottom layout
        # Process the sentence tree
        process_S(dot, parse_tree)

    else:
        raise ValueError("Only 'dependency' and 'reed-kellogg' styles are currently supported.")

    return dot.pipe(format='svg').decode('utf-8')

def process_S(dot, tree):
    """Process a sentence (S) node, handling subject and predicate."""
    np = [child for child in tree if child.label() == 'NP'][0]  # Subject
    vp = [child for child in tree if child.label() == 'VP'][0]  # Predicate
    
    subject_id = process_NP(dot, np)
    verb_id = process_VP(dot, vp)
    
    separator_id = f"{id(tree)}_sep"
    dot.node(separator_id, '|', shape='plaintext')
    dot.edge(subject_id, separator_id, dir='none')
    dot.edge(separator_id, verb_id, dir='none')
    dot.subgraph([('rank', 'same'), (subject_id,), (separator_id,), (verb_id,)])

def process_NP(dot, tree):
    """Process a noun phrase (NP), handling the head noun and modifiers."""
    # Find the head noun (last NN)
    head = [child for child in tree if child.label().startswith('NN')][-1]
    head_label = ' '.join(head.leaves())
    head_id = f"{id(head)}"
    dot.node(head_id, head_label, shape='plaintext')
    
    # Find modifiers (e.g., adjectives labeled JJ)
    modifiers = [child for child in tree if child.label() == 'JJ']
    for mod in modifiers:
        mod_label = ' '.join(mod.leaves())
        mod_id = f"{id(mod)}"
        dot.node(mod_id, mod_label, shape='plaintext')
        dot.edge(head_id, mod_id, constraint=False, style='dashed')  # Slanted modifier line
        dot.edge(head_id, mod_id, style='invis')  # Position below
    
    return head_id

def process_VP(dot, tree):
    """Process a verb phrase (VP), handling the verb and complements."""
    # Find the verb (VB*)
    verb = [child for child in tree if child.label().startswith('VB')][0]
    verb_label = ' '.join(verb.leaves())
    verb_id = f"{id(verb)}"
    dot.node(verb_id, verb_label, shape='plaintext')
    
    # Handle complements (e.g., NP or PP)
    complements = [child for child in tree if child.label() in ['NP', 'PP']]
    for comp in complements:
        if comp.label() == 'PP':
            pp_id = process_PP(dot, comp)
            dot.edge(verb_id, pp_id, constraint=False, style='dashed')  # Connect to preposition
        elif comp.label() == 'NP':
            obj_id = process_NP(dot, comp)
            dot.edge(verb_id, obj_id, constraint=False, dir='none')  # Direct object
    
    return verb_id

def process_PP(dot, tree):
    """Process a prepositional phrase (PP), handling preposition and object."""
    prep = [child for child in tree if child.label() == 'IN'][0]  # Preposition
    prep_label = ' '.join(prep.leaves())
    prep_id = f"{id(prep)}"
    dot.node(prep_id, prep_label, shape='plaintext')
    
    obj = [child for child in tree if child.label() == 'NP'][0]  # Object of preposition
    obj_id = process_NP(dot, obj)
    
    dot.edge(prep_id, obj_id, dir='none')
    dot.subgraph([('rank', 'same'), (prep_id,), (obj_id,)])
    
    return prep_id