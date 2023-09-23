# Detection_faux_billets
Ce répertoire contient une implémentation pour la détection de faux billets. Le code utilise des méthodes de visualisation, une régression logistique et une méthode de clustering K-Means pour classifier et distinguer entre de vrais billets et de faux billets.

## Contenu
- **Notebook_Detection.py**: Fichier principal contenant toutes les fonctions et implémentations pour la détection.

## Fonctionnalités

### Visualisation
- Des graphiques pour visualiser les marges et les longueurs des billets réels et faux.
- Graphique pairplot pour montrer les relations entre les différentes caractéristiques des billets.

### Régression logistique
- Implémentation d'une régression logistique pour prédire la véracité des billets basée sur leurs caractéristiques.
- Affichage des probabilités prédites et des prédictions finales.

### K-Means clustering
- Utilisation de la méthode K-Means pour classer les billets en deux clusters.
- Projection des individus sur le premier plan factoriel à l'aide de PCA.

### Comparatif des algorithmes
- Comparaison des performances de la régression logistique et du K-Means en utilisant la précision, le rappel et la précision.

## Installation et exécution
Clonez le répertoire sur votre machine locale.
Assurez-vous d'avoir toutes les dépendances nécessaires installées (matplotlib, seaborn, statsmodels, sklearn).
Exécutez Notebook_Detection.py.

