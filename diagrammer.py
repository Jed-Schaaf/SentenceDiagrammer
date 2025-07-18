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
        return dot.pipe(format='svg').decode('utf-8')
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
        return svg_buffer.getvalue()
    else:
        raise ValueError("Only 'dependency' and 'reed-kellogg' styles are currently supported.")


def draw_line(ax, start_pos, end_pos, line_style='solid', color='black'):
    """
    Draw a line on the Matplotlib axis between two points.
    Parameters:
    - ax: Matplotlib axis object
    - start_pos: Starting coordinates of the line
    - end_pos: Ending coordinates of the line
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


def draw_text(ax, text, x, y, ha='left', va='bottom', rotate=False, head_y=None):
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
    #text_ax = ax.text(x, y, text, ha=ha, va=va, rotation=-45 if rotate else 0)
    width = 10#text_ax.figure.get_figwidth()
    height = 10#text_ax.figure.get_figheight()
    # Calculate line from text to head's baseline
    theta = np.radians(45)
    x1 = x - width / 2
    y1 = y + 1
    x2 = x + width / 2
    y2 = y + 1
    if rotate and head_y is not None:
        # Rotate endpoints
        x1_rot = x #+ (x1 - x) * np.cos(theta) - (y1 - y) * np.sin(theta)
        y1_rot = head_y #y1 + (x1 - x) * np.sin(theta) + (y1 - y) * np.cos(theta)
        x2_rot = x + width#(x2 - x) * np.cos(theta) - (y2 - y) * np.sin(theta)
        y2_rot = head_y + height #y2 + (x2 - x) * np.sin(theta) + (y2 - y) * np.cos(theta)
        # Extend line to head_y
        m = (y2_rot - y1_rot) / (x2_rot - x1_rot) if x2_rot != x1_rot else float('inf')
        if m != float('inf') and m != 0:
            intersect_x = x1_rot + (head_y - y1_rot) / m
            draw_line(ax, [intersect_x, head_y], [x2_rot, y2_rot])
            ax.text((intersect_x + x2_rot) / 2 + 1,
                    (head_y + y2_rot) / 2,
                    text, ha=ha, va=va, rotation=-45)
            width = x2_rot - intersect_x # Update actual dimensions
        else:
            ax.text((x1_rot + x2_rot) / 2 + 1,
                    (head_y + y2_rot) / 2,
                    text, ha=ha, va=va, rotation=-45)
            draw_line(ax, [x1_rot, head_y], [x1_rot, y1_rot])
            draw_line(ax, [x1_rot, y1_rot], [x2_rot, y2_rot])
            width = x2_rot - x1_rot # Update actual dimensions
        height = y2_rot - head_y # Update actual dimensions
    else:
        ax.text(x, y, text, ha=ha, va=va, rotation=0)
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
    if n == 0:
        return []
    spacing_per_modifier = 10
    total_width = max(head_width, n * spacing_per_modifier)
    baseline_start = x #if is_subject else x - total_width / 2
    baseline_end = baseline_start + total_width
    attachment_xs = np.linspace(baseline_start, baseline_end, n + 2)[1:-1]
    slant_length = 10
    for i, mod in enumerate(modifiers):
        mod_label = ' '.join(mod.leaves())
        mod_x = attachment_xs[i]
        text_x = mod_x #- slant_length
        text_y = y + slant_length
        draw_text(ax, mod_label, text_x, text_y, rotate=True, head_y=head_y)
    return attachment_xs


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
    #print_sentence_tree(ax, tree, 0, 40)
    # Handle coordinated sentences (e.g., S -> S CC S)
    if any(child.label() == 'CC' for child in tree):
        s_nodes = [child for child in tree if child.label() == 'S']
        cc = [child for child in tree if child.label() == 'CC'][0]
        cc_label = ' '.join(cc.leaves())
        total_width = 0
        total_height = 0
        for i, s_node in enumerate(s_nodes):
            width, height = process_s(ax, s_node, x, y + i*30)
            total_width = max(total_width, width)
            total_height += height
            if i < len(s_nodes) - 1:
                w, h = draw_text(ax, cc_label, x, y + i*30 + 15)
                draw_line(ax, [x - w/2, y + i*30], [x - w/2, y + i*30 + 15], line_style='dashed')
                draw_line(ax, [x + w/2, y + i*30 + 15], [x + w/2, y + i*30 + 30], line_style='dashed')
        return total_width, total_height
    else:
        np = [child for child in tree if child.label() == 'NP'][0]
        vp = [child for child in tree if child.label() == 'VP'][0]
        subject_width, subject_height = process_np(ax, np, x, y, is_subject=True)
        separator_x = x + subject_width + 2
        draw_line(ax, [separator_x, y + 3], [separator_x, y - 6])
        draw_line(ax, [separator_x-2, y], [separator_x+2, y])
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
        total_width = 0
        n_nodes = len(np_nodes)
        total_height = n_nodes * 8
        for i, np_node in enumerate(np_nodes):
            width, height = process_np(ax, np_node, x, y + i*8 - total_height/2, is_subject)
            total_width = max(total_width, width)
            if i < len(np_nodes) - 1:
                w, _ = draw_text(ax, cc_label, x, y + i*8 - total_height/2)
                draw_line(ax, [x - w/2, y + i*8 - total_height/2], [x - w/2, y + i*8 - total_height/2 + 4], line_style='dashed')
                draw_line(ax, [x + w/2, y + i*8 - total_height/2 + 4], [x + w/2, y + i*8 - total_height/2 + 8], line_style='dashed')
        return total_width, total_height
    # Find head (noun or pronoun)
    # Fallback to tree root if no noun/pronoun found
    head = next((child for child in tree if child.label().startswith('NN') or child.label() in ['PRP', 'PRP$']), tree)
    head_label = ' '.join(head.leaves())
    head_width = 10#len(head_label) * 1.25  # Approximate width

    # Process modifiers (determiners and adjectives)
    modifier_labels = ['DT', 'PDT', 'WDT', 'CD', 'JJ']
    attachment_xs = process_modifiers(ax, tree, x, y, head_width, y, modifier_labels, is_subject)

    if len(attachment_xs) != 0:
        baseline_start = min(attachment_xs) #if is_subject else x - head_width / 2
        baseline_end = max(attachment_xs)
    else:
        baseline_start = x #- head_width / 2
        baseline_end = x + head_width #/ 2

    pps = [child for child in tree if child.label() == 'PP']
    pp_x = baseline_end + 2
    for pp in pps:
        pp_width, _ = process_pp(ax, pp, pp_x, y, head_y=y)
        pp_x += pp_width + 2
    total_width = pp_x - x if pps else baseline_end - x
    baseline_end = x + total_width

    head_x = x + total_width / 2
    draw_text(ax, head_label, head_x, y, rotate=False)
    draw_line(ax, [baseline_start, y], [baseline_end, y])

    return total_width, 10


def process_vp(ax, tree, x, y):
    """Process a verb phrase (VP), handling verb, auxiliaries, particles, and complements."""
    # Handle auxiliary verbs (MD, VB*) and main verb
    verbs = [child for child in tree if child.label().startswith('VB') or child.label() == 'MD']
    verb_label = ' '.join(' '.join(v.leaves()) for v in verbs)
    verb_width = 10#len(verb_label) * 1.25

    modifier_labels = ['RB','ADVP']
    attachment_xs = process_modifiers(ax, tree, x, y, verb_width, y, modifier_labels)

    if len(attachment_xs) != 0:
        verb_baseline_start = x
        verb_baseline_end = max(attachment_xs) + 10
    else:
        verb_baseline_start = x #- verb_width / 2
        verb_baseline_end = x + verb_width #/ 2

    # Process objects and complements (NP, PP, ADJP)
    complements = [child for child in tree if child.label() in ['NP', 'PP', 'ADJP']]
    comp_x = verb_baseline_end + 2
    objects = [child for child in complements if child.label() == 'NP']
    if len(objects) == 2:
        indirect_obj, direct_obj = objects
        process_indirect_object(ax, indirect_obj, comp_x, y, head_y=y)
        comp_x += 10
        draw_line(ax, [comp_x, y], [comp_x+4, y])
        draw_line(ax, [comp_x+2, y-6], [comp_x+2, y])
        comp_x += 4
        do_width, _ = process_np(ax, direct_obj, comp_x, y)
        comp_x += do_width
        for comp in [comp for comp in complements if comp not in objects]:
            if comp.label() == 'PP':
                pp_width, _ = process_pp(ax, comp, comp_x, y, head_y=y)
                comp_x += pp_width
    else:
        for comp in complements:
            separator_x = comp_x
            comp_x += 2
            draw_line(ax, [separator_x-2, y-6], [separator_x+2, y])
            comp_width = 0
            if comp.label() == 'NP':
                comp_width, _ = process_np(ax, comp, comp_x, y)
            elif comp.label() == 'PP':
                comp_width, _ = process_pp(ax, comp, comp_x, y-10, head_y=y-10)
            elif comp.label() == 'ADJP':
                comp_width, _ = process_adjp(ax, comp, comp_x, y)
            comp_x += comp_width + 2

    verb_baseline_end = comp_x
    verb_x = (verb_baseline_start + verb_baseline_end) / 2
    draw_text(ax, verb_label, verb_x, y, rotate=False)
    draw_line(ax, [verb_baseline_start, y], [verb_baseline_end, y])

    return comp_x - x, 10

def process_pp(ax, tree, x, y, head_y):
    """Process a prepositional phrase (PP), handling preposition and object."""
    prep = [child for child in tree if child.label() == 'IN'][0]
    prep_label = ' '.join(prep.leaves())
    slant_length = 10
    prep_x = x + slant_length/2
    prep_y = y + slant_length
    draw_text(ax, prep_label, prep_x, prep_y, rotate=True, head_y=head_y)
    obj = [child for child in tree if child.label() == 'NP'][0]
    obj_x = x + slant_length
    obj_width, _ = process_np(ax, obj, obj_x, y)
    draw_line(ax, [prep_x, prep_y], [obj_x, y], line_style='dashed')
    return 20, 10#obj_width + 5, 10

def process_indirect_object(ax, tree, x, y, head_y):
    """Process an indirect object."""
    slant_length = 10
    io_x = x + slant_length
    io_y = y + slant_length
    draw_line(ax, [x, head_y], [io_x, io_y])
    return process_np(ax, tree, io_x, io_y)

def process_adjp(ax, tree, x, y):
    """Process an adjective phrase (ADJP), handling adjectives and adverbs."""
    head = [child for child in tree if child.label() == 'JJ'][-1]
    head_label = ' '.join(head.leaves())
    head_width = 10#len(head_label) * 0.75  # Approximate width
    # Process adverbs modifying adjectives
    modifier_labels = ['RB']
    attachment_xs = process_modifiers(ax, tree, x, y, head_width, y, modifier_labels)
    # Extend adjective baseline based on intersection points
    if len(attachment_xs) != 0:
        baseline_start = x #- head_width / 2
        baseline_end = max(attachment_xs)
    else:
        baseline_start = x #- head_width / 2
        baseline_end = x + head_width #/ 2

    head_x = (baseline_start + baseline_end) / 2
    draw_text(ax, head_label, head_x, y, rotate=False)
    draw_line(ax, [baseline_start, y], [baseline_end, y])
    return head_width, 10


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
        svg_doc = generate_diagram(
            #"He walked and I rode and she spun.",
            #"She baked me a cake.",
            "The very big red cat quickly chased the small gray mouse quietly into the hole.",
            'reed-kellogg')
        of.write(svg_doc)