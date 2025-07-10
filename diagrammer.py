from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from nltk import Tree
from nlp_module import parse_sentence


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
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)
        ax.axis('off')
        process_s(ax, parse_tree, x=20, y=20)
        plt.gca().invert_yaxis()  # Invert y-axis to match top-to-bottom layout
        svg_buffer = StringIO()
        plt.savefig(svg_buffer, format='svg')
        plt.close(fig)
        retval = svg_buffer.getvalue()
    else:
        raise ValueError("Only 'dependency' and 'reed-kellogg' styles are currently supported.")
    return retval


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
    start_x, start_y = start_pos
    end_x, end_y = end_pos
    ax.plot([start_x, end_x], [start_y, end_y], linestyle=linestyle, color=color)


def draw_text(ax, text, x, y, ha='center', va='bottom', rotate=False, head_y=None):
    """
    Draw text on the Matplotlib axis at specified coordinates.
    Parameters:
    - ax: Matplotlib axis object
    - text: String to draw
    - x, y: Coordinates for text placement
    - ha: Horizontal alignment (default: 'center')
    - va: Vertical alignment (default: 'bottom')
    - rotation: Rotation angle in degrees (default: 0)
    - head_y: Y coordinate of the head word's baseline (default: None)
    """
    text_ax = ax.text(x, y, text, ha=ha, va=va, rotation=-45 if rotate else 0)
    width = text_ax.figure.get_figwidth()
    height = text_ax.figure.get_figheight()
    # Calculate line from text to head's baseline
    theta = np.radians(45)
    x1 = x - width / 2
    y1 = y + 1
    x2 = x + width / 2
    y2 = y + 1
    if rotate and head_y is not None:
        # Rotate endpoints
        x1_rot = x + (x1 - x) * np.cos(theta) - (y1 - y) * np.sin(theta)
        y1_rot = y1 + (x1 - x) * np.sin(theta) + (y1 - y) * np.cos(theta)
        x2_rot = x + (x2 - x) * np.cos(theta) - (y2 - y) * np.sin(theta)
        y2_rot = y2 + (x2 - x) * np.sin(theta) + (y2 - y) * np.cos(theta)
        # Extend line to head_y
        m = (y2_rot - y1_rot) / (x2_rot - x1_rot) if x2_rot != x1_rot else float('inf')
        if m != float('inf') and m != 0:
            intersect_x = x1_rot + (head_y - y1_rot) / m
            draw_line(ax, [intersect_x, head_y], [x2_rot, y2_rot])
            width = x2_rot - intersect_x # Update actual dimensions
        else:
            draw_line(ax, [x1_rot, head_y], [x1_rot, y1_rot])
            draw_line(ax, [x1_rot, y1_rot], [x2_rot, y2_rot])
            width = x2_rot - x1_rot # Update actual dimensions
        height = y2_rot - head_y # Update actual dimensions
    else:
        draw_line(ax, [x1, y], [x2, y])
    return width, height


def process_modifiers(ax, tree, x, y, head_width, head_y, modifier_labels, is_subject=False):
    """
    Process and draw modifiers for a head word in a Reed-Kellogg diagram.
    Parameters:
    - ax: Matplotlib axis object
    - tree: Parse tree node containing modifiers
    - x, y: Starting coordinates for the head word
    - head_width: Width of the head word
    - head_y: Y-coordinate of the head word's baseline
    - modifier_labels: List of labels for modifiers to process (e.g., ['JJ'] for adjectives)
    - is_subject: Boolean indicating if the head word is a subject (affects modifier positioning)
    Returns:
    - List of intersection x-coordinates for extending the head word's baseline
    """
    modifiers = [child for child in tree if child.label() in modifier_labels]
    n = len(modifiers)
    intersection_xs = []
    if n == 0:
        return intersection_xs
    slant_length = 10
    mod_labels = [' '.join(mod.leaves()) for mod in modifiers]
    mod_widths = [len(label) * 0.75 for label in mod_labels]  # Approximate width
    total_mod_width = sum(mod_widths) #+ 3 * (n - 1)  # 3 is spacing between modifiers
    if is_subject:
        start_x = x - total_mod_width / 2  # Extend left for subjects
    else:
        start_x = x + head_width / 2  # Start from head's center for non-subjects
    current_x = start_x
    for mod_label, mod_width in zip(mod_labels, mod_widths):
        text_x = current_x + mod_width / 2
        text_y = y + slant_length
        # Get intersection for baseline extension
        mod_width, _ = draw_text(ax, mod_label, text_x, text_y, rotate=True, head_y=head_y)
        intersect_x = text_x + mod_width
        intersection_xs.append(intersect_x)
        current_x += mod_width #+ 3
    return min(intersection_xs), max(intersection_xs)

def print_sentence_tree(ax, tree, x, y):
    """displays sentence tree for debugging purposes"""
    cur_y = y
    if not isinstance(tree, str):
        cur_y = y + 4
        draw_text(ax, tree.label() + '->' + ' '.join(tree.leaves()), x, cur_y, ha='left')
        for node in tree:
            cur_y = print_sentence_tree(ax, node, x+3, cur_y)
    return cur_y


def process_s(ax, tree, x, y):
    """Process a sentence (S) node, handling subject, predicate, and coordinated/subordinate clauses."""
    print_sentence_tree(ax, tree, 0, 40)
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
        draw_line(ax, [separator_x, y + 3], [separator_x, y - 6])  # Vertical separator
        predicate_x = separator_x + 2
        predicate_width, predicate_height = process_vp(ax, vp, predicate_x, y)
        # Handle subordinate clauses (SBAR)
        sbar_nodes = [child for child in tree if child.label() == 'SBAR']
        if sbar_nodes:
            sbar_x = predicate_x + predicate_width + 2
            for sbar in sbar_nodes:
                sbar_width, sbar_height = process_sbar(ax, sbar, sbar_x, y + 20)
                #draw_line(ax, [predicate_x + predicate_width / 2, y], [sbar_x, y + 20], line_style='dashed')
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
            total_width += width + 3
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
    # Process modifiers (determiners and adjectives)
    modifier_labels = ['DT', 'PDT', 'WDT', 'CD', 'JJ']
    intersection_xs = process_modifiers(ax, tree, x, y, head_width, y, modifier_labels, is_subject)
    # Extend head baseline based on intersection points
    if intersection_xs:
        if is_subject:
            baseline_start_x = intersection_xs[0] # Extend left for subjects
            baseline_end_x = x + head_width / 2
        else:
            baseline_start_x = x - head_width / 2 # Start from head's center for non-subjects
            baseline_end_x = intersection_xs[1]
    else:
        baseline_start_x = x - head_width / 2
        baseline_end_x = x + head_width / 2
    y_line = y
    draw_line(ax, [baseline_start_x, y_line], [baseline_end_x, y_line])
    return head_width, 10


def process_vp(ax, tree, x, y):
    """Process a verb phrase (VP), handling verb, auxiliaries, particles, and complements."""
    # Handle auxiliary verbs (MD, VB*) and main verb
    verbs = [child for child in tree if child.label().startswith('VB') or child.label() == 'MD']
    particles = [child for child in tree if child.label() == 'RP']
    if not verbs:
        verbs = tree
    verb_label = ' '.join(' '.join(v.leaves()) for v in verbs + particles)
    verb_width = len(verb_label) * 1.5  # Approximate width
    draw_text(ax, verb_label, x + verb_width / 2, y)
    # Process adverbs
    modifier_labels = ['RB']#['ADVP','RB']
    intersection_xs = process_modifiers(ax, tree, x, y, verb_width, y, modifier_labels)
    # Extend verb baseline based on intersection points
    if intersection_xs:
        baseline_start_x = x - verb_width / 2
        baseline_end_x = intersection_xs[1]
    else:
        baseline_start_x = x - verb_width / 2
        baseline_end_x = x + verb_width / 2
    y_line = y
    draw_line(ax, [baseline_start_x, y_line], [baseline_end_x, y_line])
    # Process complements (NP, PP, ADJP)
    complements = [child for child in tree if child.label() in ['NP', 'PP', 'ADJP']]
    comp_x = x + verb_width + 2
    for comp in complements:
        if comp.label() == 'NP':
            obj_width, obj_height = process_np(ax, comp, comp_x, y)
            separator_x = comp_x - 1
            draw_line(ax, [separator_x, y], [separator_x, y - 6])  # Vertical separator
            comp_x += obj_width + 2
        elif comp.label() == 'PP':
            pp_width, pp_height = process_pp(ax, comp, comp_x, y)
            draw_line(ax, [x + verb_width / 2, y], [x + verb_width / 2 - 6, y - 6])
            comp_x += pp_width + 2
        elif comp.label() == 'ADJP':
            adjp_width, adjp_height = process_adjp(ax, comp, comp_x, y)
            draw_line(ax, [x + verb_width / 2, y], [x + verb_width / 2 - 6, y - 6])
            draw_text(ax, '/', x + verb_width / 2 + 1, y)
            comp_x += adjp_width + 2
    return verb_width, 10  # Approximate height


def process_pp(ax, tree, x, y):
    """Process a prepositional phrase (PP), handling preposition and object."""
    prep = [child for child in tree if child.label() == 'IN'][0]
    prep_label = ' '.join(prep.leaves())
    w, h = draw_text(ax, prep_label, x, y+10, rotate=True, head_y=y)
    obj = [child for child in tree if child.label() == 'NP'][0]
    obj_x = x + w
    obj_width, obj_height = process_np(ax, obj, obj_x, y+h)
    return obj_x + obj_width - x, 10  # Approximate width and height


def process_adjp(ax, tree, x, y):
    """Process an adjective phrase (ADJP), handling adjectives and adverbs."""
    head = [child for child in tree if child.label() == 'JJ'][-1]
    if not head:
        head = tree
    head_label = ' '.join(head.leaves())
    head_width = len(head_label) * 1.5  # Approximate width
    draw_text(ax, head_label, x + head_width / 2, y)
    # Process adverbs modifying adjectives
    modifier_labels = ['RB']
    intersection_xs = process_modifiers(ax, tree, x, y, head_width, y, modifier_labels)
    # Extend adjective baseline based on intersection points
    if intersection_xs:
        baseline_start_x = x - head_width / 2
        baseline_end_x = intersection_xs[1]
    else:
        baseline_start_x = x - head_width / 2
        baseline_end_x = x + head_width / 2
    y_line = y
    draw_line(ax, [baseline_start_x, y_line], [baseline_end_x, y_line])
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
        svg_doc = generate_diagram("The very big red cat quickly chased the small gray mouse quietly into the hole.", 'reed-kellogg')
        of.write(svg_doc)