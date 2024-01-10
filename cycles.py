from solvepl import edges_containing_node
import networkx as nx
import pulp as pl
import time
import copy


def destruct_cycles(graph, time_limit, path_to_cplex):
    """
       Résout le programme linéaire à base de cycles

       @param graph: Le graphe d'origine.
       @param time_limit: Limite de temps pour la résolution du problème.
       @param path_to_cplex: Chemin vers CPLEX.
       @return: La variables de décision obtenue (x), l'objectif obtenue et le graphe obtenue.
       """

    res_graph = copy.deepcopy(graph)

    solver = pl.CPLEX_CMD(path=path_to_cplex, timeLimit=time_limit, logPath="info.log", msg=False)
    model = pl.LpProblem("main_problem", pl.LpMinimize)

    # Nombre de sommet
    nb_nodes = res_graph.number_of_nodes()
    cycles = nx.cycle_basis(res_graph)
    edges = res_graph.edges

    # Création des variables
    x = {e: pl.LpVariable(cat=pl.LpBinary, name="x_{0}".format(e)) for e in edges}
    x.update({(j, i): pl.LpVariable(cat=pl.LpBinary, name="x_{0}".format((j, i))) for (i, j) in edges})
    y = {v: pl.LpVariable(cat=pl.LpBinary, name="y_{0}".format(v)) for v in range(1, nb_nodes + 1)}

    # Création de la fonction objective
    model += pl.lpSum(y[i] for i in range(1, nb_nodes + 1))

    # Création des contraintes
    # Contrainte (3)
    model += pl.lpSum(x[e] for e in edges) == nb_nodes - 1

    for (i, j) in edges:
        model += x[(i, j)] == x[(j, i)]

    # Contrainte (4)
    for cycle in cycles:
        long_cycle = len(cycle)
        model += pl.lpSum(x[(cycle[k], cycle[(k + 1) % long_cycle])] for k in range(long_cycle)) <= long_cycle - 1

    # Contrainte (5)
    for v in range(1, nb_nodes + 1):
        model += pl.lpSum(x[e] for e in edges_containing_node(res_graph, v)) - 2 <= res_graph.degree[v] * y[v]
        model.solve(solver)

    model.solve(solver)

    model.writeLP("model.lp")

    for edge in edges:
        if not x[edge].value():
            res_graph.remove_edge(edge[0], edge[1])

    return x, pl.value(model.objective), res_graph


def link_components(original_graph, pl_graph):
    """
       Rélie les composantes connexes du graphe

       @param graph: Le graphe d'origine.
       @return: Le graphe obtenue après ajout des arêtes entre les paires de noeuds qui sont dans des composantes différentes
    """
    original_edges = original_graph.edges
    res_graph = copy.deepcopy(pl_graph)
    connected_components = list(nx.connected_components(res_graph))
    components_number = len(connected_components)

    for n1 in range(components_number - 1):
        comp1 = connected_components[n1]
        for n2 in range(n1 + 1, components_number):
            comp2 = connected_components[n2]
            res_graph.add_edges_from([(node1, node2) for node1 in comp1 for node2 in comp2
                                      if (node1, node2) in original_edges or (node2, node1) in original_edges])

    return res_graph


def solve_by_cycles(graph, time_limit, path_to_cplex):
    """
       Résout le problème MBVST à base de cycles

       @param graph: Le graphe d'origine.
       @param time_limit: Limite de temps pour la résolution du problème.
       @param path_to_cplex: Chemin vers CPLEX.
       @return: La variables de décision (x), l'objectif et le graphe obtenues.
       """

    start_time = time.time()

    connex_graph = copy.deepcopy(graph)
    x, z, pl_graph = destruct_cycles(connex_graph, 20, path_to_cplex)
    connected = nx.is_connected(pl_graph)
    if not connected:
        connex_graph = link_components(connex_graph, pl_graph)

    while not connected and (time.time() - start_time < time_limit):
        x, z, pl_graph = destruct_cycles(connex_graph, 20, path_to_cplex)
        connected = nx.is_connected(pl_graph)
        if not connected:
            connex_graph = link_components(connex_graph, pl_graph)

    return x, z, connex_graph