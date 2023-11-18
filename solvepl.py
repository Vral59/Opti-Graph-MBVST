import pulp as pl
from itertools import chain, combinations
import networkx as nx


def powerset(iterable):
    s = list(iterable)
    return [list(subset) for subset in chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]


def edges_in_subset(graph, subset):
    subgraph = graph.subgraph(subset)
    subset_edges = list(subgraph.edges())
    original_edges = list(graph.edges())

    # Inversion des arêtes qui ne sont pas présentes dans le graphe original
    for i, edge in enumerate(subset_edges):
        if edge not in original_edges:
            subset_edges[i] = (edge[1], edge[0])

    return subset_edges


def edges_containing_node(graph, node):
    node_edges = [edge for edge in graph.edges() if node in edge]
    return node_edges


def pl_expo(graph, time_limit, path_to_cplex):

    solver = pl.CPLEX_CMD(path=path_to_cplex, timeLimit=time_limit, logPath="info.log")
    model = pl.LpProblem("main_problem", pl.LpMinimize)

    all_subsets = list(powerset(list(range(1, graph.number_of_nodes()+1))))
    all_subsets = all_subsets[graph.number_of_nodes()+1:]
    print(all_subsets)
    # Création des variables
    x = {e: pl.LpVariable(cat=pl.LpBinary, name="x_{0}".format(e)) for e in graph.edges}
    y = {i: pl.LpVariable(cat=pl.LpBinary, name="y_{0}".format(i)) for i in range(1, graph.number_of_nodes()+1)}

    # Création de la fonction objective
    model += pl.lpSum(y[i] for i in range(1, graph.number_of_nodes()+1))

    # Création des contraintes
    # Contrainte (3)
    model += pl.lpSum(x[e] for e in graph.edges) == graph.number_of_nodes() - 1

    # Contrainte (4)
    for S in all_subsets:
        model += pl.lpSum(x[e] for e in edges_in_subset(graph, S)) <= len(S) - 1

    # Contrainte (5)
    for i in range(1, graph.number_of_nodes() + 1):
        model += pl.lpSum(x[e] for e in edges_containing_node(graph, i)) - 2 <= graph.degree[i]*y[i]
        model.solve(solver)

    model.solve(solver)

    model.writeLP("model.lp")

    return x, y
