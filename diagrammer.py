from graphviz import Digraph
from nltk import Tree
from nlp_module import parse_sentence


def generate_diagram(sentence, style='dependency'):
    """
    Generate an SVG diagram from the parsed sentence structure.
    Supports 'dependency' and 'reed-kellogg' styles.
    """
    doc = parse_sentence(sentence=sentence)
    sent = list(doc.sents)[0]
    parse_tree = Tree.fromstring(getattr(getattr(sent, '_'), 'parse_string'))
    dot = Digraph()

    if style == 'dependency':
        for token in doc:
            dot.node(str(token.i), token.text)
            if token.head != token:
                dot.edge(str(token.head.i), str(token.i), label=token.dep_)
    elif style == 'reed-kellogg':
        dot.attr(rankdir='LR')  # Left-to-right layout for horizontal alignment
        process_s(dot, parse_tree)
    else:
        raise ValueError("Only 'dependency' and 'reed-kellogg' styles are currently supported.")

    return dot.pipe(format='svg').decode('utf-8')


def process_s(dot, tree):
    """Process a sentence (S) node, handling subject and predicate."""
    np = [child for child in tree if child.label() == 'NP'][0]  # Subject
    vp = [child for child in tree if child.label() == 'VP'][0]  # Predicate

    subject_id = process_np(dot, np, is_subject=True)
    predicate_id = process_vp(dot, vp)

    separator_id = f"{id(tree)}_sep"
    dot.node(separator_id, '|', shape='plaintext')

    # Align subject, separator, and predicate horizontally
    dot.edge(subject_id, separator_id, style='invis')
    dot.edge(separator_id, predicate_id, style='invis')
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node(subject_id)
        s.node(separator_id)
        s.node(predicate_id)


def process_np(dot, tree, is_subject=False):
    """Process a noun phrase (NP), handling the head noun, modifiers, and compounds."""
    # Handle compound subjects
    if len([child for child in tree if child.label().startswith('NN')]) > 1 and is_subject:
        return process_compound(dot, tree, 'NP')

    head = [child for child in tree if child.label().startswith('NN')][-1]
    head_label = ' '.join(head.leaves())
    head_id = f"{id(head)}"
    dot.node(head_id, head_label, shape='plaintext')

    # Process adjectives (JJ)
    modifiers = [child for child in tree if child.label() == 'JJ']
    for mod in modifiers:
        mod_label = ' '.join(mod.leaves())
        mod_id = f"{id(mod)}"
        dot.node(mod_id, mod_label, shape='plaintext')
        # Slanted line: upper left to lower right
        dot.edge(head_id, mod_id, style='dashed', dir='none', constraint='false')

    return head_id


def process_vp(dot, tree):
    """Process a verb phrase (VP), handling verb, objects, and complements."""
    verb = [child for child in tree if child.label().startswith('VB')][0]
    verb_label = ' '.join(verb.leaves())
    verb_id = f"{id(verb)}"
    dot.node(verb_id, verb_label, shape='plaintext')

    # Process complements (NP, PP, or subject complement)
    complements = [child for child in tree if child.label() in ['NP', 'PP', 'ADJP']]
    for comp in complements:
        if comp.label() == 'NP':
            # Check for indirect object (simplified: second NP after verb)
            if 'dobj' in [token.dep_ for token in parse_sentence(' '.join(tree.leaves()))] and complements.index(
                    comp) == 0:
                ind_obj_id = process_np(dot, comp)
                dot.edge(verb_id, ind_obj_id, style='dashed', constraint='false')
            else:  # Direct object
                obj_id = process_np(dot, comp)
                separator_id = f"{verb_id}_{obj_id}_sep"
                dot.node(separator_id, '|', shape='plaintext')
                dot.edge(verb_id, separator_id, style='invis')
                dot.edge(separator_id, obj_id, style='invis')
                with dot.subgraph() as s:
                    s.attr(rank='same')
                    s.node(verb_id)
                    s.node(separator_id)
                    s.node(obj_id)
        elif comp.label() == 'PP':
            pp_id = process_pp(dot, comp)
            dot.edge(verb_id, pp_id, style='dashed', constraint='false')
        elif comp.label() == 'ADJP':  # Subject complement
            comp_id = process_adjp(dot, comp)
            slant_id = f"{verb_id}_{comp_id}_slant"
            dot.node(slant_id, '/', shape='plaintext')
            dot.edge(verb_id, slant_id, style='invis')
            dot.edge(slant_id, comp_id, style='invis')
            with dot.subgraph() as s:
                s.attr(rank='same')
                s.node(verb_id)
                s.node(slant_id)
                s.node(comp_id)

    # Process adverbs (RB)
    adverbs = [child for child in tree if child.label() == 'RB']
    for adv in adverbs:
        adv_label = ' '.join(adv.leaves())
        adv_id = f"{id(adv)}"
        dot.node(adv_id, adv_label, shape='plaintext')
        dot.edge(verb_id, adv_id, style='dashed', dir='none', constraint='false')

    return verb_id


def process_pp(dot, tree):
    """Process a prepositional phrase (PP), handling preposition and object."""
    prep = [child for child in tree if child.label() == 'IN'][0]
    prep_label = ' '.join(prep.leaves())
    prep_id = f"{id(prep)}"
    dot.node(prep_id, prep_label, shape='plaintext')

    obj = [child for child in tree if child.label() == 'NP'][0]
    obj_id = process_np(dot, obj)

    # Slanted preposition line, horizontal object
    dot.edge(prep_id, obj_id, style='dashed', dir='none', constraint='false')
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node(prep_id)
        s.node(obj_id)

    return prep_id


def process_adjp(dot, tree):
    """Process an adjective phrase (ADJP) as a subject complement."""
    head = [child for child in tree if child.label() == 'JJ'][-1]
    head_label = ' '.join(head.leaves())
    head_id = f"{id(head)}"
    dot.node(head_id, head_label, shape='plaintext')
    return head_id


def process_compound(dot, tree, phrase_type):
    """Process compound elements (e.g., compound subjects or predicates)."""
    heads = [child for child in tree if child.label().startswith('NN' if phrase_type == 'NP' else 'VB')]
    conj = [child for child in tree if child.label() == 'CC'][0] if 'CC' in [child.label() for child in tree] else None

    head_ids = []
    for head in heads:
        head_label = ' '.join(head.leaves())
        head_id = f"{id(head)}"
        dot.node(head_id, head_label, shape='plaintext')
        head_ids.append(head_id)

    if conj:
        conj_label = ' '.join(conj.leaves())
        conj_id = f"{id(conj)}"
        dot.node(conj_id, conj_label, shape='plaintext')

        # Split horizontal line with conjunction
        for i, head_id in enumerate(head_ids):
            if i == 0:
                dot.edge(head_id, conj_id, style='invis')
            else:
                dot.edge(conj_id, head_id, style='invis')
        with dot.subgraph() as s:
            s.attr(rank='same')
            for head_id in head_ids:
                s.node(head_id)
            s.node(conj_id)

    return head_ids[0]  # Return first head for connection purposes


if __name__ == "__main__":
    with open("test.svg", 'w') as of:
        svg_doc = generate_diagram("The big cat quickly chased the small mouse.", 'reed-kellogg')
        of.write(svg_doc)