import os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def kmeans_clustering(X, **params):
    kmeans = KMeans(**params)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def hierarchical_clustering(X, **params):
    hc = AgglomerativeClustering(**params)
    labels = hc.fit_predict(X)
    return labels, hc

def dbscan_clustering(X, **params):
    params['eps'] = float(params['eps'])
    dbscan = DBSCAN(**params)
    labels = dbscan.fit_predict(X)
    return labels, dbscan

def plot_pca_scree_plot(X, out_dir="cluster_plots"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(X)

    eigenvalues = pca.explained_variance_
    components = range(1, len(eigenvalues) + 1)

    plt.figure()
    plt.plot(components, eigenvalues, marker='o')
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    
    scree_plot_path = os.path.join(out_dir, "scree_plot.png")
    plt.savefig(scree_plot_path)
    plt.close()

def evaluate_k_means(X, base_params, k_values):
    inertias = []
    silhouettes = []
    n_samples = X.shape[0]

    for k in k_values:
        params = base_params.copy()
        params['n_clusters'] = k
        kmeans = KMeans(**params)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        inertias.append(inertia)

        if 1 < k < n_samples:
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = None
        silhouettes.append(sil_score)

    plot_pca_scree_plot(X, out_dir="cluster_plots")

    return inertias, silhouettes