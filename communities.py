from difflib import SequenceMatcher
from itertools import combinations, product

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from community import community_louvain
from plotly.subplots import make_subplots


def explore_communities(df, sample_size, similarity_threshold):
    """
    Create network graph, centralities (degree, betweenness, closeness) graph,
    community detection graph.

    Parameters
    ----------
    df : Input dataframe containing text data.
    sample_size : Variable indicating the share of dataset 
        to use for the analysis.
    similarity_threshold : Variable indicating the threshold above which
        to treat posts as similar.

    Returns
    -------
    fig1 : Plotly Figure Object.
    fig2 : Plotly Figure Object.
    """

    # Sample and clean data
    df = df.sample(n=sample_size, random_state=13).drop_duplicates(subset='TweetID')
    df['clean_text'] = df['clean_text'].astype(str)

    # Generate user combinations and compute similarities
    user_combinations = combinations(df['UserID'].unique(), 2)

    def compute_similarity(user1, user2):
        posts1 = df.loc[df['UserID'] == user1, 'clean_text']
        posts2 = df.loc[df['UserID'] == user2, 'clean_text']
        similarities = [
            SequenceMatcher(None, post1, post2).ratio()
            for post1, post2 in product(posts1, posts2)
            if SequenceMatcher(None, post1, post2).ratio() > similarity_threshold
        ]
        return sum(similarities) / len(similarities) if similarities else 0

    similarities = {
        (user1, user2): compute_similarity(user1, user2)
        for user1, user2 in user_combinations
    }

    sim_df = pd.DataFrame([
        {'user1': user1, 'user2': user2, 'value': value}
        for (user1, user2), value in similarities.items() if value > 0
    ])

    # Create the graph
    G = nx.from_pandas_edgelist(sim_df, source='user1', target='user2', edge_attr='value')

    # Detect communities
    communities = community_louvain.best_partition(G)
    num_communities = max(communities.values()) + 1

    # Generate positions
    pos = nx.kamada_kawai_layout(G)

    # Prepare edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        node_color.append(communities[node])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Community',
                tickvals=list(range(num_communities)),
                xanchor='left',
                titleside='right'
            ),
            color=node_color,
            colorbar_tickmode='array',
            colorbar_tickvals=list(range(num_communities)),
        )
    )

    fig1 = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         title='Network Graph',
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20, l=5, r=5, t=40),
                         xaxis=dict(showgrid=False, zeroline=False),
                         yaxis=dict(showgrid=False, zeroline=False),
                         clickmode='event+select'
                     ))

    # Calculate centralities
    centralities = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G)
    }

    # Plot centralities with Plotly
    fig2 = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=('Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality'))

    for i, (name, centrality) in enumerate(centralities.items(), start=1):
        df_centrality = pd.DataFrame.from_dict(
            centrality,
            orient='index',
            columns=['centrality']).nlargest(20, 'centrality')
        fig2.add_trace(
            go.Bar(x=df_centrality.index, y=df_centrality['centrality'], name=name),
            row=1,
            col=i
        )

    fig2.update_layout(showlegend=False, title_text="Top 20 Nodes by Centrality")

    return fig1, fig2
