"""
For graphing out decision trees
"""

import click
import graphviz

from mirrorverse.chinook.tree.run_or_drift import RunOrDriftBranch


def build_graph(decision_tree):
    """
    Input:
    - decision_tree (DecisionTree): the decision tree to graph

    Build a graph of the decision tree
    """
    dot = graphviz.Digraph(decision_tree.__name__, format="png")

    def build_node_label(root):
        states = set()
        choice_states = set()
        for builder in root.BUILDERS:
            states.update(builder.STATE)
            choice_states.update(builder.CHOICE_STATE)

        features = root.FEATURE_COLUMNS
        outcomes = root.OUTCOMES

        label = f"""<
        <FONT>{root.__name__}</FONT>
        <BR/>
        <FONT POINT-SIZE="10">States: {sorted(states)}</FONT>
        <BR/>
        <FONT POINT-SIZE="10">Choice States: {sorted(choice_states)}</FONT>
        <BR/>
        <FONT POINT-SIZE="10">Features: {sorted(features)}</FONT>
        <BR/>
        <FONT POINT-SIZE="10"><B>Outcomes:</B> {sorted(outcomes)}</FONT>
        >"""
        return label

    def build_nodes(root, dot):
        label = build_node_label(root)
        dot.node(root.__name__, label=label, shape="box")
        for choice, branch in root.BRANCHES.items():
            build_nodes(branch, dot)
            if len(root.BRANCHES) > 1:
                dot.edge(root.__name__, branch.__name__, label=choice)
            else:
                dot.edge(root.__name__, branch.__name__)

    build_nodes(decision_tree, dot)

    return dot


@click.command()
@click.option("--decision_tree", "-d", help="decision tree to graph", required=True)
def main(decision_tree):
    """
    Main function for graphing out decision trees
    """
    manifest = {
        "RunOrDriftBranch": RunOrDriftBranch,
    }
    assert decision_tree in manifest

    dot = build_graph(manifest[decision_tree])
    dot.render()
