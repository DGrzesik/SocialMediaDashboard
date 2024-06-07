import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from reduce_dimensionality import reduce_dimensionality
from text_processing import sentence_embedding


def create_scatter_plot(results: np.ndarray, text: list, target: list, color_scale: bool = True) -> go.Figure:
    """
    Create a scatter plot using Plotly.

    Parameters:
        results (np.ndarray): Array of reduced dimensionality embeddings.
        text (list): List of text labels for each data point.
        target (list): List of target labels for each data point.
        color_scale (bool): Flag indicating whether to use color scale. 
            If True, colors are determined by the target values (e.g., Sentiment). 
            If False, each category is assigned a unique color (e.g., Topic).

    Returns:
        go.Figure: Plotly Figure object.
    """
    # Initialize a new Plotly figure
    fig = go.Figure()

    if color_scale:
        # Add scatter plot with color scale based on target values
        fig.add_trace(go.Scatter(
            x=results[:, 0],
            y=results[:, 1],
            mode='markers',
            text=text,
            hoverinfo='text',
            marker=dict(
                color=target,
                colorscale='Viridis',
                colorbar=dict(title='Target')
            )
        ))
    else:
        # Map topics to colors
        colors = px.colors.qualitative.Plotly
        color_map = {category: colors[i % len(colors)] for i, category in enumerate(set(target))}

        # A set of categories already in legend
        added_to_legend = set()
        for x, y, tweet_text, category in zip(results[:, 0], results[:, 1], text, target):
            color = color_map[category]

            # Add scatter plot with unique color for each category
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                text=tweet_text,
                hoverinfo='text',
                marker=dict(color=color),
                name=category if category not in added_to_legend else None  # Add name to legend only once
            ))

            # Topics already in the legend
            added_to_legend.add(category)

        # Hide points without a name in the legend
        for trace in fig.data:
            if trace.name is None:
                trace.update(showlegend=False)

    # Update layout of the figure
    fig.update_layout(
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        width=1000,
        height=900
    )

    return fig


def generate_visualization(
        dataframe: pd.DataFrame,
        available_features: list,
        method: str,
        params: dict,
        features: str,
        target: str) -> go.Figure:
    """
    Main function that runs data processing and chart creation.

    Parameters:
        dataframe (pd.DataFrame): Input dataframe containing text data.
        available_columns (list): List of available engagement columns.
        method (str): Dimensionality reduction method.
        params (dict): Parameters for the reduction method.
        features (str): String indicating the type of features to visualize.
            If 'Text', the embeddings of the 'text' column will be visualized.
            If 'Engagement', dimensionality reduction will be performed, and the columns related to 
            engagement/interactions will be visualized.
        target (str): Target column in the dataset.

    Returns:
        go.Figure: Plotly Figure object.
    """

    # Based on selected features, either create sentence embeddings or pass columns for dimensionality reduction
    if features == 'text':
        X_features = sentence_embedding(clean_text=np.array(dataframe.clean_text))
    elif features == 'engagement':
        X_features = dataframe[list(set(available_features) - {"text"})].values
    else:
        raise ValueError(f"Unsupported feature type: {features}")
        
    # Reduce features dimensionality
    reduced_features = reduce_dimensionality(features=X_features, method=method, params=params)

    # Define color_scale variable based on target
    color_scale = False if target == 'topic' else True

    fig = create_scatter_plot(
        results=reduced_features,
        text=list(dataframe.text),
        target=list(dataframe[target]),
        color_scale=color_scale)
    return fig
