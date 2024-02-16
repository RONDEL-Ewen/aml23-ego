import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import sys

def main(features_pathfile : str, labels_pathfile :str) :

    # Charger les données à partir du fichier pkl
    with open(features_pathfile, 'rb') as file:
        data = pickle.load(file)

    with open(labels_pathfile, 'rb') as file:
        data2 = pickle.load(file)

    # Créer les DataFrames
    df = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)

    # Extraire les features et les narrations
    features = np.array([item['features_RGB'] for item in df['features']])
    narrations = np.array(df2['narration'])

    # Calculer la moyenne des features
    features_mean = np.mean(features, axis=1)

    # Réduction de la dimension avec t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=40.0,
        learning_rate=1000,
        n_iter=200000,
        random_state=42
    )
    features_reduced = tsne.fit_transform(features_mean)

    # Appliquer DBSCAN
    dbscan = DBSCAN(eps=2, min_samples=3)  # Vous pouvez ajuster les paramètres epsilon et min_samples selon votre cas d'utilisation
    labels = dbscan.fit_predict(features_reduced)

    # Créer un DataFrame pour les données réduites
    df_reduced = pd.DataFrame(features_reduced, columns=['Dim1', 'Dim2'])
    df_reduced['narrations'] = narrations
    df_reduced['cluster_labels'] = labels

    # Rassembler les labels associés à chaque cluster
    clusters = {}
    for cluster_label in np.unique(labels):
        if cluster_label == -1:
            # Cluster de bruit, ignorer
            continue
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_narrations = df_reduced.loc[cluster_indices, 'narrations']
        clusters[cluster_label] = list(set(cluster_narrations))

    # Afficher les clusters et les narrations associées
    for cluster_label, cluster_narrations in clusters.items():
        print(f"Cluster {cluster_label}:")
        for narration in cluster_narrations:
            print(f"  - {narration}")

    # Visualisation
    fig, ax = plt.subplots()
    for cluster_label, cluster_narrations in clusters.items():
        cluster_indices = np.where(labels == cluster_label)[0]
        ax.scatter(features_reduced[cluster_indices, 0], features_reduced[cluster_indices, 1], label=f'Cluster {cluster_label}')

    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('Clustering des features réduites avec DBSCAN')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


if __name__ == '__main__' :
    
    FEATURES_PATHFILE = sys.argv[1]
    LABELS_PATHFILE = sys.argv[2]

    main(FEATURES_PATHFILE, LABELS_PATHFILE)
