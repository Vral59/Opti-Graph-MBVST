# Projet de Résolution de Problème d'Arbres Optimaux

## Description

Ce projet propose différentes solutions pour résoudre le Problème MBVST à l'aide de différentes méthodes, telles que la Programmation Linéaire, l'analyse de cycles dans les graphes, et l'utilisation de modèles de Machine Learning.

## Structure du Projet

Le projet est composé des fichiers suivants :

- **main.py** : Le script principal qui coordonne l'exécution des différentes méthodes de résolution et évalue les résultats.
  
- **cycles.py** : Fournit une résolution du problème en utilisant une méthode basée sur l'analyse de la base de cycles d'un graphe.

- **solvePL.py** : Contient plusieurs fonctions utilisant la Programmation Linéaire pour résoudre le Problème d'Arbres Optimaux.

- **ml.py** : Propose des fonctions pour entraîner et appliquer différents modèles de Machine Learning afin de résoudre le problème.

- **edge_models_adaboost.joblib** et **edge_models_xgboost.joblib** : Deux modèles de Machine Learning préalablement entraînés et sauvegardés pour une utilisation ultérieure dans le code principal.

- **list_train_graph.txt** : Fichier contenant la liste des graphes utilisés pour l'entraînement des modèles. Les graphes résolus correspondants peuvent être trouvés dans le dossier **instances/Low_graph_solved**.

## Utilisation

1. **Préparation des Données d'Entraînement (optionnel) ** :
   - Utilisez la fonction `create_list_graph(graph_dic)` dans le fichier **main.py** pour générer une liste de graphes à partir d'un dossier spécifié.
   - Exécutez le script principal pour résoudre le Problème d'Arbres Optimaux sur ces graphes et enregistrer les résultats.

2. **Exécution du Projet** :
   - Modifiez le chemin du fichier à résoudre dans la fonction `main()` du fichier **main.py**.
   - Exécutez **main.py** pour résoudre le Problème d'Arbres Optimaux en utilisant différentes méthodes.

3. **Changement du Modèle de Résolution** :
   - Sélectionnez le modèle de Machine Learning à utiliser en modifiant le modèle dans `joblib.load()` dans la fonction `main()`.

## Résultats

- Les résultats de chaque méthode de résolution sont affichés, y compris les arbres optimaux générés.

- Les modèles de Machine Learning peuvent être entraînés en activant la variable `bool_train_model` et enregistrés pour une utilisation ultérieure.

- L'efficacité des modèles est évaluée en créant un arbre optimal basé sur les prédictions de probabilités.

- Les sommets de degré supérieur ou égal à trois dans l'arbre optimal sont affichés en tant que score dans les différentes méthodes.

## Remarques

- Assurez-vous d'avoir les dépendances nécessaires installées.

- Le chemin vers l'exécutable CPLEX doit être spécifié dans la variable `PATH_TO_CPLEX` dans **main.py**.
