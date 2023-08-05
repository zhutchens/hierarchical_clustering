from anytree import Node, RenderTree
import graphviz
from IPython.display import display, Image


def check_for_cycles(lesser_word, higher_word, ontological_hierarchy):
    """Check if adding the word pair (lesser_word, higher_word) to the ontology
    would create a cycle."""
    while True:
        parent = find_parent(higher_word, ontological_hierarchy)
        if parent is None:
            return False
        if parent == lesser_word:
            return True
        higher_word = parent


def find_parent(word, ontological_hierarchy):
    """Find the parent of a word in the ontology."""
    for parent, children in ontological_hierarchy.items():
        if word in children:
            return parent
    return None


def construct_ontology_hierarchy(
    ordered_directed_relations: list,
):
    """Construct the ontology hierarchy from the ordered directed relations.

    Parameters
    ----------
    ordered_directed_relations : list
        The ordered directed relations.

    Returns
    -------
    ontological_hierarchy : dict
        The ontology as a dictionary of parent words and their children.
    words_with_parents : set
        The set of words that have a parent.
    """
    ontological_hierarchy = {}

    words_with_parents = set()

    for relation in ordered_directed_relations:
        if relation[0] == relation[1]:
            continue
        if check_for_cycles(relation[1], relation[0], ontological_hierarchy):
            continue

        if relation[1] in words_with_parents:
            continue
        if relation[0] in ontological_hierarchy:
            ontological_hierarchy[relation[0]].append(relation[1])
            words_with_parents.add(relation[1])
        else:
            ontological_hierarchy[relation[0]] = [relation[1]]
            words_with_parents.add(relation[1])

    # Order the ontology with respect to the number of children.
    ontological_hierarchy = dict(
        sorted(
            ontological_hierarchy.items(), key=lambda item: len(item[1]), reverse=True
        )
    )

    return ontological_hierarchy, words_with_parents


def print_hierarchy_tree_from_ontology(
    ontological_hierarchy: dict,
    words_with_parents: set,
):
    """Print the ontology as a tree.

    Parameters
    ----------
    ontological_hierarchy : dict
        The ontology as a dictionary of parent words and their children.
    words_with_parents : set
        The set of words that have a parent.
    """
    tree = {}

    # create a node for each parent key and add it to the tree
    for parent in ontological_hierarchy:
        if parent not in tree:
            tree[parent] = Node(parent)

    # add children nodes for each child of a parent
    for parent, children in ontological_hierarchy.items():
        for child in children:
            if child not in tree:
                tree[child] = Node(child)
            tree[child].parent = tree[parent]

    parents = list(ontological_hierarchy.keys())

    # find the root node
    for parent in parents:
        if parent not in words_with_parents:
            root = parent
            # print the tree
            for pre, fill, node in RenderTree(tree[root]):
                print("%s%s" % (pre, node.name))


def draw_hierarchy_tree_from_ontology(
    ontological_hierarchy: dict,
    relations_to_verbs: dict,
    drawing_orientation: str = "TB",
    title: str = None,
    topic_modelling_chapters: list[str] = None,
):
    """Draw the ontology as a tree using Graphviz.

    Parameters
    ----------
    ontological_hierarchy : dict
        The ontology as a dictionary of parent words and their children.
    relations_to_verbs : dict
        The dictionary of relations to verbs.
    drawing_orientation : str
        The drawing orientation of the graph. Can be "TB" (top-to-bottom) or "LR" (left-to-right).
        The default is "TB".
    title : str
        The title of the graph. The default is None.
    topic_modelling_chapters : list
        The list of chapters extracted using topic modelling. The default is None.
    """
    if drawing_orientation == "TB":
        graph = graphviz.Digraph(
            graph_attr={"rankdir": "TB"},
            node_attr={"shape": "box", "fontname": "Calibri Italic", "fontsize": "12"},
            edge_attr={"fontname": "Calibri Italic", "fontsize": "12"},
        )
    elif drawing_orientation == "LR":
        graph = graphviz.Digraph(
            graph_attr={"rankdir": "LR"},
            node_attr={"shape": "box", "fontname": "Calibri Italic", "fontsize": "12"},
            edge_attr={"fontname": "Calibri Italic", "fontsize": "12"},
        )

    if topic_modelling_chapters is not None:
        if title is not None:
            # Put the title together with the chapters in the top of the graph.
            graph.attr(
                label=f"{title}\n\nChapters:\n" + "\n".join(topic_modelling_chapters),
                labelloc="t",
                labeljust="l",
                fontname="Calibri Italic",
                fontsize="12",
            )
        else:
            # Put the chapters in the top of the graph.
            graph.attr(
                label="Chapters:\n" + "\n".join(topic_modelling_chapters),
                labelloc="t",
                labeljust="l",
                fontname="Calibri Italic",
                fontsize="12",
            )

    if title is not None and topic_modelling_chapters is None:
        # Add title to the top of the graph.
        graph.attr(
            label=title,
            labelloc="t",
            labeljust="c",
            fontname="Calibri Italic",
            fontsize="20",
        )

    # Create a node for each parent key and add it to the graph.
    for parent in ontological_hierarchy:
        if parent not in graph:
            graph.node(parent, fontname="Calibri Italic", fontsize="12")

    # Add children nodes for each child of a parent.
    for parent, children in ontological_hierarchy.items():
        for child in children:
            if child not in graph:
                graph.node(child, fontname="Calibri Italic", fontsize="12")
            label = ", ".join(list(set(relations_to_verbs[(parent, child)])))
            graph.edge(
                parent, child, label=label, fontname="Calibri Italic", fontsize="12"
            )

    display(Image(graph.pipe(format="png", renderer="cairo")))
