import numpy as np
import pacmap
import trimap
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensionality(features: np.ndarray, method: str, params: dict) -> np.ndarray:
    """
    Reduce dimensionality of embeddings using the specified method.

    Parameters:
    features (np.ndarray): Array of input features.
    method (str): Dimensionality reduction method.
    params (dict): Parameters for the reduction method.

    Returns:
    np.ndarray: Reduced dimensionality array.
    """
    if method == 't-SNE':
        model = TSNE(perplexity=params['perplexity'], n_components=params['n_components'],
                     learning_rate=params['learning_rate'], init='pca', n_iter=2500, random_state=23)
    elif method == 'PCA':
        model = PCA(n_components=params['n_components'])
    elif method == 'UMAP':
        model = umap.UMAP(init='pca', n_components=params['n_components'], n_neighbors=params['n_neighbors'],
                          metric=params['metric'])
    elif method == 'PaCMAP':
        model = pacmap.PaCMAP(n_components=params['n_components'], n_neighbors=params['n_neighbors'],
                              distance=params['metric'])
    elif method == 'TriMAP':
        model = trimap.TRIMAP(n_dims=params['n_components'])

    results = model.fit_transform(features)
    return results
