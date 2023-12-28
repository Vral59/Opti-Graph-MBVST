import networkx as nx
import matplotlib.pyplot as plt
import solvepl
import os
import random
import cycles
import ml
import joblib
import time

PATH_TO_CPLEX = r'C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe'


def create_list_graph(graph_dic):
    """
    Crée une liste de 30% des graphes d'un dossier.

    :param graph_dic: Le chemin vers le dossier de graph.
    :return: None
    """
    # Obtenez la liste des fichiers dans le dossier
    fichiers_graphes = [fichier for fichier in os.listdir(graph_dic) if fichier.endswith('.txt')]

    # Calculez le nombre de fichiers à sélectionner (30%)
    nombre_fichiers_selectionnes = int(0.3 * len(fichiers_graphes))

    # Sélectionnez de manière aléatoire 30% des fichiers
    fichiers_selectionnes = random.sample(fichiers_graphes, nombre_fichiers_selectionnes)

    # Spécifiez le chemin vers le fichier texte de sortie
    fichier_sortie = 'list_train_graph.txt'

    # Enregistrez les noms des fichiers sélectionnés dans le fichier texte
    with open(fichier_sortie, 'w') as f:
        for fichier in fichiers_selectionnes:
            f.write(fichier + '\n')

    print(f"{nombre_fichiers_selectionnes} fichiers ont été sélectionnés et enregistrés dans {fichier_sortie}.")


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


def draw_tree(nb_node, x):
    """
    Dessine un graph à partir du dictionnaire des arrêtes et le renvoie

    :param nb_node: Nombre de Noeud
    :param x: Le dictionnaire
    :return: Le graph
    """
    # Création d'un graphe dirigé
    G = nx.Graph()

    # Ajout des nœuds
    G.add_nodes_from(range(1, nb_node+1))

    # Ajout des arêtes si la valeur dans le dictionnaire est proche de 1
    for edge, var in x.items():
        if 0.8 <= var.value() <= 1.2:  # Gestion de la précision numérique
            G.add_edge(edge[0], edge[1])

    return G


def main():
    """
    Fonction principale exécutant les étapes du script.
    """
    # Graph à résoudre
    file_path = 'instances/Spd_Inst_Rid_Final2/Spd_RF2_40_81_731.txt'  # Remplacez par le chemin de votre fichier
    graph = read_graph_from_file(file_path)
    directed_g = graph.to_directed()

    plt.title("Graphe de base")
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()

    time_limit = 120

    temps_debut = time.time()
    x, z = solvepl.pl_flot(directed_g, time_limit, PATH_TO_CPLEX)
    temps_fin = time.time()
    print('Score Flot: ', z)
    duree_execution = temps_fin - temps_debut
    print(f"La résolution avec flot a pris {duree_execution} secondes pour s'exécuter.")

    x, z = solvepl.pl_flot_multi(directed_g, time_limit, PATH_TO_CPLEX)
    print('Score Multi-flot: ', z)
    solved_graph = draw_tree(graph.number_of_nodes(), x)
    plt.title("Arbre résolu par multi-flot")
    nx.draw(solved_graph, with_labels=True, font_weight='bold')
    plt.show()

    x, z = solvepl.pl_martin2(graph, time_limit, PATH_TO_CPLEX)
    print('Score Martin: ', z)
    martin_graph = draw_tree(graph.number_of_nodes(), x)
    plt.title("Arbre résolu avec formulation de Martin")
    nx.draw(martin_graph, with_labels=True, font_weight='bold')
    plt.show()

    x, z, graph = cycles.solve_by_cycles(graph, time_limit, PATH_TO_CPLEX)
    print('Score Cycles: ', z)
    cycle_graph = draw_tree(graph.number_of_nodes(), x)
    plt.title("Arbre résolu par base de cycle")
    nx.draw(cycle_graph, with_labels=True, font_weight='bold')
    plt.show()

    X_graph = []
    Y_tree = []

    with open('list_train_graph.txt', 'r') as file:
        # Lire chaque ligne du fichier
        for line in file:
            # Nettoyer la ligne en enlevant les espaces blancs au début et à la fin
            filename = line.strip()
            treePath = "instances/Low_graph_solved/" + filename
            graphPath = "instances/Spd_Inst_Rid_Final2/" + filename
            xgraph = read_graph_from_file(graphPath)
            X_graph.append(xgraph)
            ytree = read_graph_from_file(treePath)
            Y_tree.append(ytree)

    bool_train_model = False
    if bool_train_model:
        # Entrainement du modèle
        edge_models = ml.train_edge_models(X_graph, Y_tree)
        joblib.dump(edge_models, 'edge_models_xgboost.joblib')
    else:
        # Utilisation d'un modèle déjà sauvegarder
        edge_models = joblib.load('edge_models_xgboost.joblib')

    print("Prédiction des proba")

    temps_debut = time.time()
    predictions = ml.predict_proba_for_new_graph(graph, edge_models)
    temps_fin = time.time()
    duree_execution = temps_fin - temps_debut
    print(f"La prédiction a pris {duree_execution} secondes pour s'exécuter.")

    # Creation de l'arbre
    print("Création du de l'arbre")
    temps_debut = time.time()
    min_degree_tree = ml.build_minimum_degree_spanning_tree(predictions)
    temps_fin = time.time()
    duree_execution = temps_fin - temps_debut
    print(f"La creation de l'arbre a pris {duree_execution} secondes pour s'exécuter.")

    plt.title("Arbre résolu avec ML")
    nx.draw(min_degree_tree, with_labels=True, font_weight='bold')
    plt.show()

    # Evaluation du nombre de sommets de degré supérieur ou égal à trois
    degrees = dict(min_degree_tree.degree())
    list_node_high_degree = [node for node, degree in degrees.items() if degree >= 3]
    print("Score Machine Learning :", len(list_node_high_degree))

    return 0


if __name__ == '__main__':
    main()
