from solvepl import edges_containing_node
import networkx as nx
import pulp as pl

def cycles_base(graph):

    cycles_nodes= list(nx.simple_cycles(graph))
    cycles_aretes = [list(zip(path, path[1:] + [path[0]])) for path in cycles_nodes]
    # Liste de sous-liste de couples (i,j): arête

    return cycles_aretes


def destruct_cycles(graph, time_limit, path_to_cplex):
    """
       Résout le problème d'optimisation MBVST avec un nombre exponentielle de contraintes.

       @param graph: Le graphe d'origine.
       @param time_limit: Limite de temps pour la résolution du problème.
       @param path_to_cplex: Chemin vers CPLEX.
       @return: Les variables de décision obtenues (x, y).
       """

    solver = pl.CPLEX_CMD(path=path_to_cplex, timeLimit=time_limit, logPath="info.log", msg=False)
    model = pl.LpProblem("main_problem", pl.LpMinimize)

    # Nombre de sommet
    nb_nodes = graph.number_of_nodes()
    cycles = cycles_base(graph)
    edges = graph.edges

    # Création des variables
    x = {e: pl.LpVariable(cat=pl.LpBinary, name="x_{0}".format(e)) for e in graph.edges}
    y = {v: pl.LpVariable(cat=pl.LpBinary, name="y_{0}".format(v)) for v in range(1, nb_nodes + 1)}

    # Création de la fonction objective
    model += pl.lpSum(y[i] for i in range(1, nb_nodes + 1))

    # Création des contraintes
    # Contrainte (3)
    model += pl.lpSum(x[e] for e in graph.edges) == nb_nodes - 1

    # Contrainte (4)
    for cycle in cycles:
        model += pl.lpSum(x[(i,j)] for (i,j) in edges if (i,j) in cycle or (j,i) in cycle) <= len(cycle) - 1

    # Contrainte (5)
    for v in range(1, nb_nodes + 1):
        model += pl.lpSum(x[e] for e in edges_containing_node(graph, v)) - 2 <= graph.degree[v] * y[v]
        model.solve(solver)

    model.solve(solver)

    model.writeLP("model.lp")

    for edge in graph.edges:
        if not x[edge].value():
            graph.remove_edge(edge[0], edge[1])


    return x, pl.value(model.objective), graph


def link_components(graph):
    connected_components = list(nx.connected_components(graph))
    components_number = len(connected_components)

    for n1 in range(components_number - 1):
        comp1 = connected_components[n1]
        for n2 in range(n1 + 1, components_number):
            comp2 = connected_components[n2]
            graph.add_edges_from([(node1, node2) for node1 in comp1 for node2 in comp2])

    return graph


def is_tree(graph):
    # Vérifie la connectivité
    is_connexe = nx.is_connected(graph)

    # Vérifie l'acyclicité
    is_acyclic = nx.is_forest(graph)  # Un graphe forestier est un graphe acyclique

    # Vérifie le nombre d'arêtes
    is_tree = is_connexe and is_acyclic and (graph.number_of_edges() == graph.number_of_nodes() - 1)

    return is_tree



def solve_by_cycles(graph, time_limit, path_to_cplex):

    while not is_tree(graph):
        x, z, graph = destruct_cycles(graph, time_limit, path_to_cplex)

        if not is_tree(graph):
            graph = link_components(graph)

    return x, z, graph