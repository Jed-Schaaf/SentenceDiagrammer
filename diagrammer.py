from io import StringIO
import matplotlib.pyplot as plt
from graphviz import Digraph
from nltk import Tree
from nlp_module import parse_sentence


def draw_text(ax, text, x, y, ha='center', va='bottom'):
    """
    Draw text on the Matplotlib axis at specified coordinates.
    Parameters:
    - ax: Matplotlib axis object
    - text: String to draw
    - x, y: Coordinates for text placement
    - ha: Horizontal alignment (default: 'center')
    - va: Vertical alignment (default: 'bottom')
    """
    text_ax = ax.text(x, y, text, ha=ha, va=va)
    width = text_ax.figure.get_figwidth()
    ax.plot([x - width / 2, x + width / 2], [y, y], linestyle='-', color='black')


def draw_line(ax, start_pos, end_pos, line_style='solid', color='black'):
    """
    Draw a line on the Matplotlib axis between two points.
    Parameters:
    - ax: Matplotlib axis object
    - start_x, start_y: Starting coordinates of the line
    - end_x, end_y: Ending coordinates of the line
    - line_style: Type of line ('solid' or 'dashed')
    - color: Line color (default: 'black')
    """
    if line_style == 'solid':
        linestyle = '-'
    elif line_style == 'dashed':
        linestyle = '--'
    else:
        raise ValueError(f"Unsupported line style: {line_style}")
    start_x = start_pos[0]
    start_y = start_pos[1]
    end_x = end_pos[0]
    end_y = end_pos[1]
    ax.plot([start_x, end_x], [start_y, end_y], linestyle=linestyle, color=color)


def generate_diagram(sentence, style='dependency'):
    """
    Generate an SVG diagram from the parsed sentence structure.
    Supports 'dependency' and 'reed-kellogg' styles.
    """
    doc = parse_sentence(sentence)

    if style == 'dependency':
        dot = Digraph()
        for token in doc:
            dot.node(str(token.i), token.text)
            if token.head != token:
                dot.edge(str(token.head.i), str(token.i), label=token.dep_)
        retval = dot.pipe(format='svg').decode('utf-8')
    elif style == 'reed-kellogg':
        sent = list(doc.sents)[0]
        parse_tree = Tree.fromstring(getattr(getattr(sent, '_'), 'parse_string'))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)
        ax.axis('off')
        process_s(ax, parse_tree, x=10, y=10)
        plt.gca().invert_yaxis()  # Invert y-axis to match top-to-bottom layout
        svg_buffer = StringIO()
        plt.savefig(svg_buffer, format='svg')
        plt.close(fig)
        retval = svg_buffer.getvalue()
    else:
        raise ValueError("Only 'dependency' and 'reed-kellogg' styles are currently supported.")

    return retval


def process_s(ax, tree, x, y):
    """Process a sentence (S) node, handling subject, predicate, and coordinated/subordinate clauses."""
    # Handle coordinated sentences (e.g., S -> S CC S)
    if any(child.label() == 'CC' for child in tree):
        s_nodes = [child for child in tree if child.label() == 'S']
        cc = [child for child in tree if child.label() == 'CC'][0]
        cc_label = ' '.join(cc.leaves())
        total_width = 0
        for i, s_node in enumerate(s_nodes):
            width, _ = process_s(ax, s_node, x + total_width, y)
            total_width += width + 10
            if i < len(s_nodes) - 1:
                draw_text(ax, cc_label, x + total_width - 5, y)
                total_width += len(cc_label) * 1.5 + 5
        return total_width, 10
    else:
        np = [child for child in tree if child.label() == 'NP'][0]  # Subject
        vp = [child for child in tree if child.label() == 'VP'][0]  # Predicate
        subject_width, subject_height = process_np(ax, np, x, y, is_subject=True)
        separator_x = x + subject_width + 2
        draw_line(ax, [separator_x, y + 2], [separator_x, y - 12])  # Vertical separator
        predicate_x = separator_x + 2
        predicate_width, predicate_height = process_vp(ax, vp, predicate_x, y)
        # Handle subordinate clauses (SBAR)
        sbar_nodes = [child for child in tree if child.label() == 'SBAR']
        if sbar_nodes:
            sbar_x = predicate_x + predicate_width + 2
            for sbar in sbar_nodes:
                sbar_width, sbar_height = process_sbar(ax, sbar, sbar_x, y + 20)
                draw_line(ax, [predicate_x + predicate_width / 2, y], [sbar_x, y + 20], line_style='dashed')
                sbar_x += sbar_width + 2
        return subject_width + predicate_width + 4, max(subject_height, predicate_height)


def process_np(ax, tree, x, y, is_subject=False):
    """Process a noun phrase (NP), handling head noun/pronoun, determiners, and modifiers."""
    # Handle coordinated NPs (e.g., NP -> NP CC NP)
    if any(child.label() == 'CC' for child in tree):
        np_nodes = [child for child in tree if child.label() == 'NP']
        cc = [child for child in tree if child.label() == 'CC'][0]
        cc_label = ' '.join(cc.leaves())
        total_width = height = 0
        for i, np_node in enumerate(np_nodes):
            width, height = process_np(ax, np_node, x + total_width, y, is_subject)
            total_width += width + 5
            if i < len(np_nodes) - 1:
                draw_text(ax, cc_label, x + total_width - 2.5, y)
                total_width += len(cc_label) * 1.5 + 5
        return total_width, height
    # Find head (noun or pronoun)
    head = None
    for child in tree:
        if child.label().startswith('NN') or child.label() in ['PRP', 'PRP$']:
            head = child
            break
    if not head:
        head = tree  # Fallback to tree root if no noun/pronoun found
    head_label = ' '.join(head.leaves())
    head_width = len(head_label) * 1.5  # Approximate width
    draw_text(ax, head_label, x + head_width / 2, y)
    # Process determiners (DT, PDT, WDT, CD) and adjectives (JJ)
    mod_y = y + 5
    mod_x = x
    for child in tree:
        if child.label() in ['DT', 'PDT', 'WDT', 'CD']:  # Determiners
            det_label = ' '.join(child.leaves())
            draw_text(ax, det_label, mod_x, mod_y)
            draw_line(ax, [x + head_width / 2, y], [mod_x, mod_y], line_style='dashed')
            mod_x += len(det_label) * 1.5 + 2
        elif child.label() == 'JJ':  # Adjectives
            adj_label = ' '.join(child.leaves())
            draw_text(ax, adj_label, mod_x, mod_y)
            draw_line(ax, [x + head_width / 2, y], [mod_x, mod_y], line_style='dashed')
            mod_x += len(adj_label) * 1.5 + 2
        elif child.label() == 'RB':  # Adverbs modifying adjectives
            adv_label = ' '.join(child.leaves())
            draw_text(ax, adv_label, mod_x, mod_y + 5)
            draw_line(ax, [mod_x + 5, mod_y + 1], [mod_x, mod_y + 5 + 1], line_style='dashed')
            mod_x += len(adv_label) * 1.5 + 2
    return head_width, 10


def process_vp(ax, tree, x, y):
    """Process a verb phrase (VP), handling verb, auxiliaries, particles, and complements."""
    # Handle auxiliary verbs (MD, VB*) and main verb
    verbs = [child for child in tree if child.label().startswith('VB') or child.label() == 'MD']
    particles = [child for child in tree if child.label() == 'RP']
    verb_label = ' '.join(' '.join(v.leaves()) for v in verbs + particles)
    verb_width = len(verb_label) * 1.5  # Approximate width
    draw_text(ax, verb_label, x + verb_width / 2, y)
    # Process complements (NP, PP, ADJP) and adverbs
    complements = [child for child in tree if child.label() in ['NP', 'PP', 'ADJP']]
    comp_x = x + verb_width + 2
    for i, comp in enumerate(complements):
        if comp.label() == 'NP':
            obj_width, obj_height = process_np(ax, comp, comp_x, y)
            separator_x = comp_x - 1
            draw_line(ax, [separator_x, y], [separator_x, y - 10])  # Vertical separator
            comp_x += obj_width + 2
        elif comp.label() == 'PP':
            pp_width, pp_height = process_pp(ax, comp, comp_x, y + 10)
            draw_line(ax, [x + verb_width / 2, y], [comp_x, y - 10], line_style='dashed')
            comp_x += pp_width + 2
        elif comp.label() == 'ADJP':
            adjp_width, adjp_height = process_adjp(ax, comp, comp_x, y)
            draw_line(ax, [x + verb_width / 2, y], [comp_x, y - 10])  # Horizontal line
            draw_text(ax, '/', x + verb_width / 2 + 1, y)
            comp_x += adjp_width + 2

    # Process adverbs (RB)
    adverbs = [child for child in tree if child.label() == 'RB']
    adv_y = y + 5
    for i, adv in enumerate(adverbs):
        adv_label = ' '.join(adv.leaves())
        adv_x = x + verb_width + i * 10
        draw_text(ax, adv_label, adv_x, adv_y)
        draw_line(ax, [x + verb_width / 2, y], [adv_x, adv_y], line_style='dashed')

    return verb_width, 10  # Approximate height


def process_pp(ax, tree, x, y):
    """Process a prepositional phrase (PP), handling preposition and object."""
    prep = [child for child in tree if child.label() == 'IN'][0]
    prep_label = ' '.join(prep.leaves())
    draw_text(ax, prep_label, x, y)

    obj = [child for child in tree if child.label() == 'NP'][0]
    obj_x = x + 5
    obj_width, obj_height = process_np(ax, obj, obj_x, y)

    # Slanted preposition line
    draw_line(ax, [x, y], [obj_x, obj_height], line_style='dashed')

    return obj_x + obj_width - x, 10  # Approximate width and height


def process_adjp(ax, tree, x, y):
    """Process an adjective phrase (ADJP), handling adjectives and adverbs."""
    head = [child for child in tree if child.label() == 'JJ'][-1]
    head_label = ' '.join(head.leaves())
    head_width = len(head_label) * 1.5  # Approximate width
    draw_text(ax, head_label, x + head_width / 2, y)
    # Process adverbs modifying adjectives
    adverbs = [child for child in tree if child.label() == 'RB']
    mod_y = y + 5
    mod_x = x
    for adv in adverbs:
        adv_label = ' '.join(adv.leaves())
        draw_text(ax, adv_label, mod_x, mod_y)
        draw_line(ax, [x + head_width / 2, y], [mod_x, mod_y], line_style='dashed')
        mod_x += len(adv_label) * 1.5 + 2
    return head_width, 10  # Approximate height


def process_sbar(ax, tree, x, y):
    """Process a subordinate clause (SBAR), handling conjunction and embedded sentence."""
    conj = [child for child in tree if child.label() == 'IN']
    conj_label = ' '.join(conj[0].leaves()) if conj else ''
    s_node = [child for child in tree if child.label() == 'S'][0]
    if conj:
        draw_text(ax, conj_label, x, y)
        s_x = x + len(conj_label) * 1.5 + 2
    else:
        s_x = x
    width, height = process_s(ax, s_node, s_x, y)
    if conj:
        draw_line(ax, [x, y], [s_x, y], line_style='dashed')
    return width + (len(conj_label) * 1.5 + 2 if conj else 0), height


if __name__ == "__main__":
    with open("test.svg", 'w') as of:
        svg_doc = generate_diagram("The big red cat quickly chased the small gray mouse.", 'reed-kellogg')
        of.write(svg_doc)