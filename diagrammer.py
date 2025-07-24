from io import StringIO
import matplotlib.pyplot as plt
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
        process_s(ax, parse_tree, x=0, y=10)
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
    #text_ax = ax.text(x, y, text, ha=ha, va=va, rotation=-45 if rotate else 0)
    width = 10#text_ax.figure.get_figwidth()
    height = 10#text_ax.figure.get_figheight()
    # Calculate line from text to head's baseline
    #theta = np.radians(45)
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
            ax.text((intersect_x + x2_rot) / 2 + 3,
                    (head_y + y2_rot) / 2 + 3,
                    text, ha=ha, va=va, rotation=-45)
            width = x2_rot - intersect_x # Update actual dimensions
        else:
            ax.text((x1_rot + x2_rot) / 2 + 3,
                    (head_y + y2_rot) / 2 + 3,
                    text, ha=ha, va=va, rotation=-45)
            draw_line(ax, [x1_rot, head_y], [x1_rot, y1_rot])
            draw_line(ax, [x1_rot, y1_rot], [x2_rot, y2_rot])
            width = x2_rot - x1_rot # Update actual dimensions
        height = y2_rot - head_y # Update actual dimensions
    else:
        ax.text(x, y, text, ha=ha, va=va, rotation=0)
        draw_line(ax, [x1, y], [x2, y])
    return width, height


def process_modifier(ax, node, x, y, head_y):
    """
    Process a single modifier node (e.g., DT, JJ, RB, PP, ADVP) in a Reed-Kellogg diagram.
    Parameters:
    - ax: Matplotlib axis object
    - node: Parse tree node for a modifier
    - x, y: Starting coordinates for the head word
    - head_y: Y-coordinate of the head word's baseline
    Returns:
    - width of the modifier(s) for extending the head word's baseline
    """
    if isinstance(node, str):
        return 0  # Skip string nodes (leaf nodes)
    if node.label() in ['DT', 'PDT', 'WDT', 'CD', 'PRP$',
                        'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
        # simple modifiers (determiners, adjectives, adverbs)
        mod_label = ' '.join(node.leaves())
        text_x = x
        text_y = y + 10
        draw_text(ax, mod_label, text_x, text_y, rotate=True, head_y=head_y)
        return 10
    elif node.label() == 'PP':
        # prepositional phrases
        prep = [child for child in node if child.label() == 'IN'][0]
        prep_label = ' '.join(prep.leaves())
        prep_x = x
        prep_y = y + 10
        draw_text(ax, prep_label, prep_x, prep_y, rotate=True, head_y=head_y)
        np = [child for child in node if child.label() == 'NP'][0]
        np_width, _ = process_np(ax, np, x+10, y+10)
        return np_width + 10  # Total width includes preposition and NP
    elif node.label() in ['ADJP', 'ADVP']:
        # adjectival and adverbial phrases
        total_width = 10
        if node.label() == 'ADJP':
            main_word = [child for child in node if child.label() == 'JJ'][0]
            adverbs = [child for child in node if child.label() == 'RB'][::-1]
        elif node.label() == 'ADVP':
            main_word = [child for child in node if child.label() == 'RB'][-1]
            adverbs = [child for child in node if child.label() == 'RB'][-2::-1]
        else:
            return 0
        adj_label = ' '.join(main_word.leaves())
        adj_x = x
        adj_y = y + 10
        draw_text(ax, adj_label, adj_x, adj_y, rotate=True, head_y=head_y)
        initial_offset = y + 10
        adv_x = adj_x + 2
        for i, adverb in enumerate(adverbs):
            adv_label = ' '.join(adverb.leaves())
            adv_y = initial_offset + i * 10
            draw_text(ax, adv_label, adv_x, adv_y, rotate=True, head_y=adv_y)
            draw_line(ax, [adv_x, adv_y], [adv_x+4, adv_y-4])
        return total_width
    return 0  # Unhandled node type


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
        subject_width, subject_height = process_np(ax, np, x, y)
        separator_x = x + subject_width + 2
        draw_line(ax, [separator_x, y + 3], [separator_x, y - 6])
        draw_line(ax, [separator_x-2, y], [separator_x+2, y])
        predicate_x = separator_x + 5
        pred_mod_width = 5
        for child in tree:
            if child.label() == 'ADVP':
                adv_width = process_modifier(ax, child, predicate_x+pred_mod_width, y, y)
                pred_mod_width += adv_width
        draw_line(ax, [separator_x, y], [predicate_x+pred_mod_width, y])
        predicate_width, predicate_height = process_vp(ax, vp, predicate_x+pred_mod_width, y)
        # Handle subordinate clauses (SBAR)
        sbar_nodes = [child for child in tree if child.label() == 'SBAR']
        if sbar_nodes:
            sbar_x = predicate_x + predicate_width + 2
            for sbar in sbar_nodes:
                sbar_width, sbar_height = process_sbar(ax, sbar, sbar_x, y + 20)
                draw_line(ax, [predicate_x + predicate_width / 2, y], [sbar_x, y + 20], line_style='dashed')
                sbar_x += sbar_width + 2
        return subject_width + predicate_width + 4, max(subject_height, predicate_height)


def process_np(ax, tree, x, y):
    """Process a noun phrase (NP), handling head noun/pronoun and all modifiers."""
    # Handle coordinated NPs (e.g., NP -> NP CC NP)
    if any(child.label() == 'CC' for child in tree):
        np_nodes = [child for child in tree if child.label() == 'NP']
        cc = [child for child in tree if child.label() == 'CC'][0]
        cc_label = ' '.join(cc.leaves())
        total_width = 0
        n_nodes = len(np_nodes)
        total_height = n_nodes * 8
        for i, np_node in enumerate(np_nodes):
            width, height = process_np(ax, np_node, x, y + i*8 - total_height/2)
            total_width = max(total_width, width)
            if i < len(np_nodes) - 1:
                w, _ = draw_text(ax, cc_label, x, y + i*8 - total_height/2)
                draw_line(ax, [x - w/2, y + i*8 - total_height/2], [x - w/2, y + i*8 - total_height/2 + 4], line_style='dashed')
                draw_line(ax, [x + w/2, y + i*8 - total_height/2 + 4], [x + w/2, y + i*8 - total_height/2 + 8], line_style='dashed')
        return total_width, total_height
    # Find head (noun or pronoun)
    head_width = 5
    head_label = ""
    head_buffer = 0
    for child in tree:
        if child.label().startswith('NN') or child.label()  == 'PRP':
            # 'NN', 'NNS', 'NNP', 'NNPS'
            head_label += ' '.join(child.leaves())
            head_width += 5
        elif child.label() == 'NP':
            width, _ = process_np(ax, child, x, y)
            head_width += width
            head_buffer = width
    # if head not found, fall back to the root
    #if not head_label:
        #head_label = ' '.join(tree.leaves())

    # Process all modifiers (DT, JJ, PP, etc.)
    total_mod_width = head_buffer
    mod_x = x+2 + head_buffer
    for child in tree:
        if child.label() in ['DT', 'PDT', 'WDT', 'CD', 'PRP$',
                             'JJ', 'JJR', 'JJS', 'PP', 'ADJP']:
            mod_width = process_modifier(ax, child, mod_x, y, y)
            total_mod_width += mod_width
            mod_x += mod_width
    
    baseline_start = x
    baseline_end = baseline_start + max(head_width, total_mod_width)
    head_x = (baseline_start + baseline_end) / 2
    draw_text(ax, head_label, head_x, y, rotate=False)
    draw_line(ax, [baseline_start, y], [baseline_end, y])
    return total_mod_width, 10

def process_vp(ax, tree, x, y):
    """Process a verb phrase (VP), handling verb, auxiliaries, particles, and complements."""
    # Handle auxiliary verbs (MD, VB*) and main verb
    verb_width = 5
    verb_label = ""
    for child in tree:
        if child.label().startswith('VB') or child.label() == 'MD':
            # 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
            verb_label += ' '.join(child.leaves())
            verb_width += 5

    # Process all modifiers (RB, ADVP, PP, etc.)
    total_mod_width = 0
    mod_x = x+5
    for child in tree:
        if child.label() in ['RB', 'RBR', 'RBS', 'ADVP', 'PP']:
            mod_width = process_modifier(ax, child, mod_x, y, y)
            total_mod_width += mod_width
            mod_x += mod_width
    
    verb_baseline_start = x
    verb_baseline_end = verb_baseline_start + max(verb_width, total_mod_width)

    # Process objects and complements (NP, ADJP)
    complements = [child for child in tree if child.label() in ['NP', 'ADJP']]
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
            if comp.label() == 'ADJP':
                comp_width, _ = process_adjp(ax, comp, comp_x, y)
                comp_x += comp_width
    else:
        for comp in complements:
            separator_x = comp_x
            comp_x += 2
            draw_line(ax, [separator_x-2, y-6], [separator_x+2, y])
            comp_width = 0
            if comp.label() == 'NP':
                comp_width, _ = process_np(ax, comp, comp_x, y)
            elif comp.label() == 'ADJP':
                comp_width, _ = process_adjp(ax, comp, comp_x, y)
            comp_x += comp_width + 2

    verb_baseline_end = comp_x
    draw_text(ax, verb_label, verb_baseline_start + verb_width / 2, y, rotate=False)
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
    adj = [child for child in tree if child.label() in ['JJ', 'JJR', 'JJS']][0]
    adj_label = ' '.join(adj.leaves())
    head_width = 10

    # Process all modifiers (RB)
    total_mod_width = 0
    mod_x = x
    for child in tree:
        if child.label() == 'RB':
            mod_width = process_modifier(ax, child, mod_x, y, y)
            total_mod_width += mod_width
            mod_x += mod_width
    
    baseline_start = x
    baseline_end = baseline_start + max(head_width, total_mod_width)
    head_x = (baseline_start + baseline_end) / 2
    draw_text(ax, adj_label, head_x, y, rotate=False)
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
            "The very big red cat quickly chased the small gray mouse into the hole in the wall very quietly.",
            #"To know him is to love him.",
            #"The little girl is very very very awful on her violin.",
            'reed-kellogg')
        of.write(svg_doc)