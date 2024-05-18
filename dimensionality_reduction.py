import pandas as pd
import numpy as np
import re   
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import trimap
import pacmap
import plotly.graph_objects as go
import nltk

# check session variable?
nltk.download("stopwords")
STOP_WORDS = nltk.corpus.stopwords.words('english')

def clean(text: str) -> str:
    """Text cleaning."""
    text = re.sub(r'https?:\/\/[^\s]+', '', text)   # remove URL
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)      # remove user tags
    text = re.sub(r'[^\w\s]', '', text)             # remove special characters
    text = re.sub('RT','', text)                    # remove RT string
    text = re.sub('rsysadmin', '', text)            # remove a special string
    text = text.lower()                             # convert to lowercase
    return text

def remove_stopwords(text: str) -> str:
    """Remove english stopwords"""
    words = text.split()
    cleaned_words = [word for word in words if word not in STOP_WORDS]
    cleaned_text = " ".join(cleaned_words)
    return cleaned_text

def sentence_embedding(clean_text: np.array) -> np.ndarray:
    """Create sentence embeddings using SentenceTransformer model."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(clean_text)
    return np.array(sentence_embeddings)

def tsne_reduction(embeddings: np.ndarray, perplexity: int, n_components: int, learning_rate: float) -> np.ndarray:
    """Dimensionality reduction using t-SNE."""
    model = TSNE(perplexity=perplexity, n_components=n_components, learning_rate=learning_rate, init='pca', n_iter=2500, random_state=23)
    results = model.fit_transform(embeddings)
    return results

def pca_reduction(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Dimensionality reduction using PCA."""
    model = PCA(n_components=n_components)
    results = model.fit_transform(embeddings)
    return results

def umap_reduction(embeddings: np.ndarray, n_components: int, n_neighbors: int, metric: str) -> np.ndarray:
    """Dimensionality reduction using UMAP."""
    model = umap.UMAP(init='pca', n_components=n_components, n_neighbors=n_neighbors, metric=metric)
    results = model.fit_transform(embeddings)
    return results

def pacmap_reduction(embeddings: np.ndarray, n_components: int, n_neighbors: int, metric: float) -> np.ndarray:
    """Dimensionality reduction using PaCMAP."""
    model = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, distance=metric)
    results = model.fit_transform(embeddings)
    return results

def trimap_reduction(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Dimensionality reduction using TriMAP."""
    model = trimap.TRIMAP(n_dims=n_components)
    results = model.fit_transform(embeddings)
    return results


def create_plot(results: np.ndarray, text: list, target: np.ndarray) -> go.Figure:
    """Create Plotly chart."""
    fig = go.Figure()

    for i in range(len(results)):
        x, y = results[i]
        tweet_text = text[i]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', text=tweet_text, hoverinfo='text'))

    fig.update_layout(title='TSNE title', xaxis_title='TSNE X', yaxis_title='TSNE Y')

    return fig

def run(dataset, method: str, params: dict, target: str) -> go.Figure:
    """Main function that runs data processing and chart creation."""
    # to be changed
    DATA_PATH = 'files/tweets-engagement-metrics.csv'
    df = pd.read_csv(DATA_PATH)
    df = df[df['Lang'] == 'en']
    df = df.head(70)

    # clean text
    df['clean_text'] = df['text'].apply(clean)
    df['clean_text'] = df['clean_text'].apply(remove_stopwords)
    
    # embedding
    sentence_embeddings = sentence_embedding(np.array(df.clean_text))

    # dimensionality reduction based on method
    if method == 't-SNE':
        results = tsne_reduction(sentence_embeddings, params['perplexity'], params['n_components'], params['learning_rate'])
    elif method == 'PCA':
        results = pca_reduction(sentence_embeddings, params['n_components'])
    elif method == 'UMAP':
        results = umap_reduction(sentence_embeddings, params['n_components'], params['n_neighbors'], params['metric'])
    elif method == 'PaCMAP':
        results = pacmap_reduction(sentence_embeddings, params['n_components'], params['n_neighbors'], params['metric'])
    elif method == 'TriMAP':
        results = trimap_reduction(sentence_embeddings, params['n_components'])

    fig = create_plot(results, list(df.text), df[target])
    return fig

# if __name__ == "__main__":
#     run()



