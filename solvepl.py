import pulp as pl
from itertools import chain, combinations
import networkx as nx


def powerset(iterable):
    """
    Retourne une liste de tous les sous-ensembles de l'itérable donné.

    @param iterable: L'itérable pour lequel générer les sous-ensembles.
    @return: Une liste de listes représentant les sous-ensembles.
    """
    s = list(iterable)
    return [list(subset) for subset in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))]


def edges_in_subset(graph, subset):
    """
    Retourne la liste des arêtes dans un sous-ensemble du graphe.

    @param graph: Le graphe d'origine.
    @param subset: Le sous-ensemble de sommets pour lequel récupérer les arêtes.
    @return: Une liste d'arêtes du sous-ensemble.
    """
    subgraph = graph.subgraph(subset)
    subset_edges = list(subgraph.edges())
    original_edges = list(graph.edges())

    # Inversion des arêtes qui ne sont pas présentes dans le graphe original
    for i, edge in enumerate(subset_edges):
        if edge not in original_edges:
            subset_edges[i] = (edge[1], edge[0])

    return subset_edges


def edges_containing_node(graph, node):
    """
    Retourne la liste des arêtes contenant un nœud spécifié dans le graphe.

    @param graph: Le graphe d'origine.
    @param node: Le nœud pour lequel récupérer les arêtes.
    @return: Une liste d'arêtes contenant le nœud.
    """
    node_edges = [edge for edge in graph.edges() if node in edge]
    return node_edges


def pl_expo(graph, time_limit, path_to_cplex):
    """
    Résout le problème d'optimisation MBVST avec un nombre exponentielle de contraintes.

    @param graph: Le graphe d'origine.
    @param time_limit: Limite de temps pour la résolution du problème.
    @param path_to_cplex: Chemin vers CPLEX.
    @return: Les variables de décision obtenues (x, y).
    """
    solver = pl.CPLEX_CMD(path=path_to_cplex, timeLimit=time_limit, logPath="info.log")
    model = pl.LpProblem("main_problem", pl.LpMinimize)

    # Nombre de sommet
    nb_nodes = graph.number_of_nodes()

    all_subsets = list(powerset(list(range(1, nb_nodes + 1))))
    all_subsets = all_subsets[nb_nodes + 1:]
    print(all_subsets)
    # Création des variables
    x = {e: pl.LpVariable(cat=pl.LpBinary, name="x_{0}".format(e)) for e in graph.edges}
    y = {v: pl.LpVariable(cat=pl.LpBinary, name="y_{0}".format(v)) for v in range(1, nb_nodes + 1)}

    # Création de la fonction objective
    model += pl.lpSum(y[i] for i in range(1, nb_nodes + 1))

    # Création des contraintes
    # Contrainte (3)
    model += pl.lpSum(x[e] for e in graph.edges) == nb_nodes - 1

    # Contrainte (4)
    for S in all_subsets:
        model += pl.lpSum(x[e] for e in edges_in_subset(graph, S)) <= len(S) - 1

    # Contrainte (5)
    for v in range(1, nb_nodes + 1):
        model += pl.lpSum(x[e] for e in edges_containing_node(graph, v)) - 2 <= graph.degree[v] * y[v]
        model.solve(solver)

    model.solve(solver)

    model.writeLP("model.lp")

    return x, y


def pl_flot(graph, time_limit, path_to_cplex):
    """
    Résout le problème MBVST avec du flot sur un graphe orienté avec la méthode de PuLP et CPLEX.

    @param graph: Le graphe orienté d'origine.
    @param time_limit: Limite de temps pour la résolution du problème.
    @param path_to_cplex: Chemin vers CPLEX.
    @return: Les variables de décision obtenues (x, y).
    """
    solver = pl.CPLEX_CMD(path=path_to_cplex, timeLimit=time_limit, logPath="info.log")
    model = pl.LpProblem("main_problem", pl.LpMinimize)

    # Sommet source
    s = 1
    # Nombre de sommet
    nb_nodes = graph.number_of_nodes()

    # Création des variables
    x = {e: pl.LpVariable(cat=pl.LpBinary, name="x_{0}".format(e)) for e in graph.edges}
    y = {v: pl.LpVariable(cat=pl.LpBinary, name="y_{0}".format(v)) for v in range(1, nb_nodes + 1)}
    f = {e: pl.LpVariable(cat=pl.LpContinuous, name="f_{0}".format(e)) for e in graph.edges}

    # Création de la fonction objective
    model += pl.lpSum(y[i] for i in range(1, nb_nodes + 1))

    # Contrainte (9)
    for v in range(1, nb_nodes + 1):
        model += pl.lpSum(x[e] for e in graph.in_edges(v)) == 1

    # Contrainte (10)
    model += pl.lpSum(f[e] for e in graph.out_edges(s)) - pl.lpSum(f[e] for e in graph.in_edges(s)) \
             == nb_nodes - 1

    # Contrainte (11)
    for v in range(1, nb_nodes + 1):
        if v != s:
            model += pl.lpSum(f[e] for e in graph.out_edges(v)) \
                     - pl.lpSum(f[e] for e in graph.in_edges(v)) == -1

    # Contrainte (12)
    for e in graph.edges:
        model += x[e] <= f[e] <= nb_nodes * x[e]

    # Constrainte (13)
    for v in range(1, nb_nodes + 1):
        model += pl.lpSum(x[e] for e in graph.out_edges(v)) + pl.lpSum(x[e] for e in graph.in_edges(v)) - 2 \
                 <= graph.degree[v]*y[v]

    # Contrainte (16)
    for e in graph.edges:
        model += f[e] >= 0

    model.solve(solver)
    model.writeLP("model.lp")

    return x, y
