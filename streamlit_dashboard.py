import streamlit as st

import constants
import data_processing
import visualization_utils

st.set_page_config(page_title="Social Media Dashboard", layout="wide")  # wide


def df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')


@st.cache_data()
def set_params(nc, p, nn, lr, m):
    return {
        'n_components': nc,
        'perplexity': p,
        'learning_rate': lr,
        'n_neighbors': nn,
        'metric': m
    }


@st.cache_data()
def get_available_features(dataframe):
    return data_processing.get_available_features(dataframe)


@st.cache_data()
def get_available_targets(dataframe):
    return data_processing.get_available_targets(dataframe)


@st.cache_data(show_spinner=False)
def get_data(datafiles):
    return data_processing.get_data(datafiles)


st.sidebar.title("Options")
st.sidebar.subheader('Upload data')

st.title("Social Media Interaction Analysis")

uploader_container = st.sidebar.container()

uploaded_files = st.sidebar.file_uploader("Upload data for analysis:", type=[".csv", ".xlsx"],
                                          accept_multiple_files=True)

clear_button = uploader_container.button("Clear memorized uploader data")

if clear_button:
    st.cache_data.clear()

with st.spinner("Validating data"):
    df = get_data(uploaded_files)

if df is not None:

    csv = df_to_csv(df)

    st.sidebar.download_button(
        label="Download cleaned data as CSV",
        data=csv,
        file_name='cleaned-data.csv',
        mime='text/csv'
    )

    show_df = st.sidebar.checkbox("Show DataFrame")

    if show_df:
        row_amount = st.sidebar.slider("Number of rows", 1, 100, 5)
        st_df = st.dataframe(df[:row_amount])

    st.sidebar.subheader('Data')

    available_features = get_available_features(df)
    available_targets = get_available_targets(df)
    selectable_features = []
    if 'text' in available_features:
        selectable_features.append('Text')
    if len(set(available_features) - {'text'}) > 0:
        selectable_features.append('Engagement')
    features = st.sidebar.selectbox('Select data to analyze', selectable_features)

    target = st.sidebar.selectbox('Select target', available_targets)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    st.sidebar.subheader('Visualization')

    method = st.sidebar.selectbox('Select visualization method', constants.METHOD_NAMES)

    st.subheader(target + ' analysis')

    st.subheader(method)

    n_components = None
    perplexity = None
    learning_rate = None
    n_neighbors = None
    metric = None

    if method == 't-SNE':
        n_components = st.sidebar.number_input("Set number of components", 2, 3, 2)
        perplexity = st.sidebar.slider("Set perplexity", 5, 50, 30)
        set_learning_rate = st.sidebar.checkbox("Set learning rate", False)
        if set_learning_rate is True:
            learning_rate = st.sidebar.slider("Set learning rate", 10, 1000, 500)
        else:
            learning_rate = 'auto'
            st.sidebar.info("Learning rate is set to 'auto'.")

    if method == 'PCA':
        n_components = st.sidebar.number_input("Set number of components", 2, 3, 2)

    if method == 'UMAP' or method == 'PaCMAP':
        n_components = st.sidebar.number_input("Set number of components", 2, 3, 2)
        n_neighbors = st.sidebar.number_input("Set number of neighbors", 1, 100, 10)
        metric = st.sidebar.selectbox("Set metric", constants.METRICS)

    if method == 'TriMAP':
        n_components = st.sidebar.number_input("Set number of components", 2, 100, 2)

    params = set_params(nc=n_components, p=perplexity, nn=n_neighbors, lr=learning_rate, m=metric)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    st.sidebar.subheader("Additional")

    show_correlation_matrix = st.sidebar.checkbox("Show correlation matrix", False)
    # st.write(df.shape)
    # for param in params.keys():
    #     value = params[param]
    #     if value is not None:
    #         st.write(param + ": " + f"{params[param] if params[param] is not None else ''}")
    visualize_button = st.button("Visualize")
    if visualize_button:
        with st.spinner('Preparing plot'):
            fig = visualization_utils.generate_visualization(df, method, params, features, target)
        st.plotly_chart(fig)

    st.markdown('<hr>', unsafe_allow_html=True)

    if show_correlation_matrix is True:
        st.subheader("Correlation matrix")
        cols = list(set(available_features) - {'text'})
        corr_df = df[cols]
        corr_matrix = corr_df.corr()

        st.dataframe(corr_matrix, width=800)

else:
    st.info("Upload files with appropriate data to see possible options!")
