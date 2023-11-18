import networkx as nx
import matplotlib.pyplot as plt
import solvepl

PATH_TO_CPLEX = r'C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe'


def read_graph_from_file(file_path):
    """
    Lit un graphe depuis un fichier et retourne l'objet graph correspondant.

    @param file_path: Chemin du fichier contenant les informations du graphe.
    @return: Un objet NetworkX représentant le graphe lu depuis le fichier.
    """
    g = nx.Graph()
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Lire le nombre de sommets depuis la première ligne
        num_vertices = int(lines[0].split()[0])

        # Ajouter les sommets au graphe
        for i in range(1, num_vertices + 1):
            g.add_node(i)

        # Ajouter les arêtes au graphe
        for line in lines[1:]:
            edge = list(map(int, line.split()[:2]))
            g.add_edge(edge[0], edge[1])

    return g


def main():
    """
    Fonction principale exécutant les étapes principales du script.
    """
    file_path = 'instances/Spd_Inst_Rid_Final2/Spd_RF2_20_27_211.txt'  # Remplacez par le chemin de votre fichier
    graph = read_graph_from_file(file_path)
    directed_g = graph.to_directed()

    print("Liste des arêtes du graphe :", graph.edges())
    print("Liste des arêtes du graphe :", directed_g.edges())

    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()

    nx.draw(directed_g, with_labels=True, font_weight='bold')
    plt.show()

    time_limit = 60

    # x, y = solvepl.pl_expo(graph, time_limit, PATH_TO_CPLEX)
    #
    # # Afficher les valeurs de y
    # for i, var in y.items():
    #     print(f"{var.name} = {var.value()}")
    #
    # for i, var in x.items():
    #     print(f"{var.name} = {var.value()}")

    x, y = solvepl.pl_flot(directed_g, time_limit, PATH_TO_CPLEX)

    # Afficher les valeurs de y
    for i, var in y.items():
        print(f"{var.name} = {var.value()}")

    for i, var in x.items():
        print(f"{var.name} = {var.value()}")

    x, y = solvepl.pl_flot_multi(directed_g, time_limit, PATH_TO_CPLEX)

    # Afficher les valeurs de y
    for i, var in y.items():
        print(f"{var.name} = {var.value()}")

    for i, var in x.items():
        print(f"{var.name} = {var.value()}")

    x, y = solvepl.pl_martin(directed_g, time_limit, PATH_TO_CPLEX)

    # Afficher les valeurs de y
    for i, var in y.items():
        print(f"{var.name} = {var.value()}")

    for i, var in x.items():
        print(f"{var.name} = {var.value()}")

    return 0


if __name__ == '__main__':
    main()
