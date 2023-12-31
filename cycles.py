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

    
    solver = pl.CPLEX_CMD(path=path_to_cplex, timeLimit=time_limit, logPath="info.log", msg=False)
    model = pl.LpProblem("main_problem", pl.LpMinimize)

    # Nombre de sommet
    nb_nodes = graph.number_of_nodes()
    cycles = nx.cycle_basis(graph)
    edges = graph.edges

    # Création des variables
    x = {e: pl.LpVariable(cat=pl.LpBinary, name="x_{0}".format(e)) for e in edges}
    x.update({(j,i): pl.LpVariable(cat=pl.LpBinary, name="x_{0}".format((j,i))) for (i,j) in edges})
    y = {v: pl.LpVariable(cat=pl.LpBinary, name="y_{0}".format(v)) for v in range(1, nb_nodes + 1)}

    # Création de la fonction objective
    model += pl.lpSum(y[i] for i in range(1, nb_nodes + 1))

    # Création des contraintes
    # Contrainte (3)
    model += pl.lpSum(x[e] for e in edges) == nb_nodes - 1

    for (i,j) in edges:
        model += x[(i,j)] == x[(j,i)]
        
    # Contrainte (4)
    for cycle in cycles:
        long_cycle = len(cycle)
        model += pl.lpSum(x[(cycle[k],cycle[(k+1)%long_cycle])] for k in range(long_cycle)) == long_cycle - 1

    # Contrainte (5)
    for v in range(1, nb_nodes + 1):
        model += pl.lpSum(x[e] for e in edges_containing_node(graph, v)) - 2 <= graph.degree[v] * y[v]
        model.solve(solver)

    model.solve(solver)

    model.writeLP("model.lp")

    for edge in edges:
        if not x[edge].value():
            graph.remove_edge(edge[0], edge[1])


    return x, pl.value(model.objective), graph



def link_components(original_graph, graph):
    """
       Rélie les composantes connexes du graphe

       @param graph: Le graphe d'origine.
       @return: Le graphe obtenue après ajout des arêtes entre les paires de noeuds qui sont dans des composantes différentes
    """
    original_edges = original_graph.edges
    connected_components = list(nx.connected_components(graph))
    components_number = len(connected_components)

    for n1 in range(components_number - 1):
        comp1 = connected_components[n1]
        for n2 in range(n1 + 1, components_number):
            comp2 = connected_components[n2]
            graph.add_edges_from([(node1, node2) for node1 in comp1 for node2 in comp2 
                        if (node1,node2) in original_edges or (node2,node1) in original_edges])

    return graph




def solve_by_cycles(graph, time_limit, path_to_cplex):
    """
       Résout le problème MBVST à base de cycles

       @param graph: Le graphe d'origine.
       @param time_limit: Limite de temps pour la résolution du problème.
       @param path_to_cplex: Chemin vers CPLEX.
       @return: La variables de décision (x), l'objectif et le graphe obtenues.
       """
    graph_copy = copy.deepcopy(graph)
    start_time = time.time()
    x, z, graph_copy = destruct_cycles(graph_copy, time_limit, path_to_cplex)
    connected = nx.is_connected(graph_copy)
    
    while not connected and (time.time()-start_time < time_limit):
        x, z, graph_copy = destruct_cycles(graph_copy, time_limit, path_to_cplex)
        connected = nx.is_connected(graph_copy)
        if not connected:
            graph_copy = link_components(graph, graph_copy)

    return x, z, graph_copy
