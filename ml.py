import numpy as np
import networkx as nx
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def calculate_global_graph_features(graph):
    """
    Calcule les mesures globales du graphe et les stocke dans un dictionnaire.

    :param graph: Le graphe d'origine.
    :return: Un dictionnaire contenant les mesures globales du graphe.
    """
    global_features = {}

    # Calcul des mesures de centralité
    global_features['closeness_centrality'] = nx.closeness_centrality(graph)
    global_features['betweenness_centrality'] = nx.betweenness_centrality(graph)

    # Informations globales sur le graphe
    global_features['number_of_nodes'] = graph.number_of_nodes()
    global_features['number_of_edges'] = graph.number_of_edges()
    global_features['radius'] = nx.radius(graph)
    global_features['diameter'] = nx.diameter(graph)
    global_features['density'] = nx.density(graph)
    global_features['average_clustering'] = nx.average_clustering(graph)

    return global_features


def edge_to_features(graph, edge, global_features_dict):
    """
    Convertit une arête d'un graphe en un ensemble de caractéristiques.

    :param graph: Le graphe d'origine.
    :param edge: L'arête du graphe.
    :param global_features_dict: Le dictionnaire contenant toutes les informations du graph.
    :return: Un ensemble de caractéristiques pour l'arête.
    """
    # Récupérer les nœuds connectés par l'arête
    node1, node2 = edge

    # Exemple de caractéristiques :
    features = []

    # 1. Degré des nœuds connectés
    features.append(graph.degree(node1))
    features.append(graph.degree(node2))

    # Bonus. Centralité des nœuds
    features.append(nx.degree_centrality(graph)[node1])
    features.append(nx.degree_centrality(graph)[node2])

    # Utilisation du dictionnaire global_features_dict
    closeness_centrality_dict = global_features_dict['closeness_centrality']
    features.append(closeness_centrality_dict[node1])
    features.append(closeness_centrality_dict[node2])

    betweenness_centrality_dict = global_features_dict['betweenness_centrality']
    features.append(betweenness_centrality_dict[node1])
    features.append(betweenness_centrality_dict[node2])

    # Bonus. Mesures de clustering
    features.append(nx.clustering(graph, node1))
    features.append(nx.clustering(graph, node2))

    # 3. Informations globales sur le graphe
    features.append(global_features_dict['number_of_nodes'])
    features.append(global_features_dict['number_of_edges'])
    features.append(global_features_dict['radius'])
    features.append(global_features_dict['diameter'])
    features.append(global_features_dict['density'])
    features.append(global_features_dict['average_clustering'])

    return features


def train_edge_models(X_graph, Y_tree):
    """
    Entraîne un modèle de classification pour prédire les arêtes dans un arbre optimal.

    :param X_graph: Liste de graphes d'entraînement.
    :param Y_tree: Liste des arbres optimaux correspondants.
    :return: Le modèle entraîné.
    """
    Y_bool = []
    X_features = []
    cpt = 0
    for graph, optimal_tree in zip(X_graph, Y_tree):
        print("New Graph : ", cpt)
        cpt += 1
        global_features_dict = calculate_global_graph_features(graph)
        for edge in graph.edges():
            features = edge_to_features(graph, edge, global_features_dict)
            X_features.append(features)
            if edge in optimal_tree.edges():
                Y_bool.append(1)
            else:
                Y_bool.append(0)

    # edge_models = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    edge_models = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    model = edge_models.fit(np.array(X_features), np.array(Y_bool))

    return model


def train_edge_models_grid(X_graph, Y_tree):
    """
    Entraîne un modèle de classification (XGBoost) en utilisant la recherche d'hyperparamètres.

    :param X_graph: Liste de graphes d'entraînement.
    :param Y_tree: Liste des arbres optimaux correspondants.
    :return: Le modèle entraîné.
    """
    graph_features_dicts = []

    # Pré-calcule les caractéristiques globales pour chaque graphe
    for graph in X_graph:
        global_features_dict = calculate_global_graph_features(graph)
        graph_features_dicts.append(global_features_dict)

    # Créez un DataFrame pour stocker les caractéristiques et les étiquettes
    import pandas as pd
    df = pd.DataFrame(columns=["features", "label"])

    # Construisez le DataFrame en utilisant les caractéristiques et les étiquettes
    # Créez une liste pour stocker les données
    data_list = []

    # Construisez la liste de données en utilisant les caractéristiques et les étiquettes
    for graph, optimal_tree, global_features_dict in zip(X_graph, Y_tree, graph_features_dicts):
        print("New Graph")
        for edge in graph.edges():
            features = edge_to_features(graph, edge, global_features_dict)
            data_list.append({"features": features, "label": 1 if edge in optimal_tree.edges() else 0})

    # Créez le DataFrame une seule fois après la boucle
    df = pd.DataFrame(data_list)
    # Divisez le DataFrame en caractéristiques (X) et étiquettes (y)
    X = np.array(df["features"].tolist())
    y = np.array(df["label"].tolist())

    # Définissez les hyperparamètres que vous souhaitez rechercher
    param_grid = {
        'n_estimators': [90, 100, 110],
        'learning_rate': [0.1, 0.15],
        'max_depth': [4, 5, 6],
        'min_child_weight': [2, 3, 4],
        'subsample': [1.0],
        'colsample_bytree': [1.0],
        'gamma': [0.1, 0.2, 0.3],
        # Ajoutez d'autres hyperparamètres à rechercher
    }

    # Initialisez le classificateur XGBoost
    xgb = XGBClassifier(random_state=42)

    # Utilisez la recherche en grille pour trouver les meilleurs hyperparamètres
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    # Obtenez le modèle avec les meilleurs hyperparamètres
    best_model = grid_search.best_estimator_

    # Affichez les meilleurs hyperparamètres
    print("Meilleurs hyperparamètres:")
    print(grid_search.best_params_)

    return best_model

def predict_proba_for_new_graph(graph, edge_models):
    """
    Prédit les probabilités d'inclusion des arêtes dans un arbre optimal pour un nouveau graphe.

    :param graph: Le nouveau graphe à évaluer.
    :param edge_models: Le modèle de classification entraîné.
    :return: Un dictionnaire des probabilités pour chaque arête du graphe.
    """
    probas = {}
    global_features_dict = calculate_global_graph_features(graph)
    for edge in graph.edges():
        features = edge_to_features(graph, edge, global_features_dict)
        proba = edge_models.predict_proba(np.array(features).reshape(1, -1))
        probas[edge] = float(proba[:, 1])

    return probas


def build_minimum_degree_spanning_tree(probabilities):
    """
    Construit un arbre couvrant de degré minimum en utilisant les probabilités d'inclusion des arêtes.

    :param probabilities: Les probabilités d'inclusion des arêtes dans l'arbre optimal.
    :return: Un arbre couvrant de degré minimum.
    """
    # Triez les arêtes par probabilité décroissante
    sorted_edges = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    # Initialisez un ensemble vide pour représenter l'arbre couvrant
    min_degree_spanning_tree = nx.Graph()

    # Parcourez les arêtes triées et ajoutez-les à l'ensemble si elles ne créent pas de cycle
    for edge, proba in sorted_edges:
        min_degree_spanning_tree.add_edge(*edge)

        # Vérifiez s'il y a un cycle après l'ajout de l'arête
        try:
            # find_cycle renverra une exception si aucun cycle n'est trouvé
            nx.find_cycle(min_degree_spanning_tree, orientation='ignore')
        except nx.NetworkXNoCycle:
            continue
        # Si l'arête crée un cycle, retirez-la
        min_degree_spanning_tree.remove_edge(*edge)

    return min_degree_spanning_tree
