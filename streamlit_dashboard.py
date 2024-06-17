import streamlit as st

import communities
import constants
import data_processing
import visualization_utils

st.set_page_config(page_title="Social Media Dashboard", layout="wide")  # wide


def df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')


@st.cache_data()
def set_params(p, nn, lr, m):
    return {
        'n_components': 2,
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
    st.session_state.show_first_plot = False
    st.session_state.show_second_plot = False
    return data_processing.get_data(datafiles)


@st.cache_data(show_spinner=False)
def visualize_data(data, a_f, m, p, f, t):
    return visualization_utils.generate_visualization(data, a_f, m, p, f, t)


@st.cache_data(show_spinner=False)
def explore_communities(data, s_s, s_t):
    return communities.explore_communities(data, s_s, s_t)


st.sidebar.title("Options")
st.sidebar.subheader('Upload data')

st.title("Social Media Interaction Analysis")

uploader_container = st.sidebar.container()

uploaded_files = st.sidebar.file_uploader("Upload data for analysis:", type=[".csv", ".xlsx"],
                                          accept_multiple_files=True)

clear_button = uploader_container.button("Clear memorized data")

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
        selectable_features.append('text')
    if len(set(available_features) - {'text'}) > 0:
        selectable_features.append('engagement')

    feature_info = st.sidebar.checkbox("Show detailed info about available features")
    if feature_info:
        st.sidebar.info("**Available features:**\n"
                        + "\n".join([f"- {feature}" for feature in available_features])
                        + "\n\n**out of:**\n"
                        + "\n".join([f"- {feature}" for feature in constants.ENGAGEMENT_FEATURES + ['text']])
                        )

    features = st.sidebar.selectbox('Select data to analyze', selectable_features)

    target = st.sidebar.selectbox('Select target', available_targets)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    st.sidebar.subheader('Visualization')

    method = st.sidebar.selectbox('Select visualization method', constants.METHOD_NAMES)

    if target == 'text':
        st.subheader('Text' + ' analysis')
    elif target == 'sentiment':
        st.subheader('Sentiment' + ' analysis')

    st.subheader(method)

    perplexity = None
    learning_rate = None
    n_neighbors = None
    metric = None

    if method == 't-SNE':
        st.sidebar.info("Reducing dimensionality to 2 components")
        perplexity = st.sidebar.slider("Set perplexity", 5, 50, 30)
        set_learning_rate = st.sidebar.checkbox("Set learning rate", False)
        if set_learning_rate is True:
            learning_rate = st.sidebar.slider("Set learning rate", 10, 1000, 500)
        else:
            learning_rate = 'auto'
            st.sidebar.info("Learning rate is set to 'auto'.")

    if method == 'PCA':
        st.sidebar.info("Reducing dimensionality to 2 components")

    if method == 'UMAP' or method == 'PaCMAP':
        st.sidebar.info("Reducing dimensionality to 2 components")
        n_neighbors = st.sidebar.number_input("Set number of neighbors", 1, 100, 10)
        metric = st.sidebar.selectbox("Set metric", constants.METRICS)

    if method == 'TriMAP':
        st.sidebar.info("Reducing dimensionality to 2 components")

    params = set_params(p=perplexity, nn=n_neighbors, lr=learning_rate, m=metric)

    if 'show_first_plot' not in st.session_state:
        st.session_state.show_first_plot = False
    if 'show_second_plot' not in st.session_state:
        st.session_state.show_second_plot = False

    button_container_vis = st.container()
    col1, col2 = button_container_vis.columns([0.08, 0.9], gap="small")
    visualize_button = col1.button("Visualize")
    hide_button = col2.button("Hide visualization")
    if visualize_button:
        st.session_state.show_first_plot = True
    if hide_button:
        st.session_state.show_first_plot = False

    if st.session_state.show_first_plot:
        with (st.spinner('Preparing plot')):
            fig = visualize_data(df, available_features, method, params, features, target)
            st.plotly_chart(fig)

    st.markdown('<hr>', unsafe_allow_html=True)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    st.sidebar.subheader("Parameters for community exploration")

    sample_size = st.sidebar.number_input("Set sample size", 0, df.shape[0], 100)
    similarity_threshold = st.sidebar.slider("Set similarity threshold", 0.0, 1.0, 0.5)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    st.subheader('Community detection')

    if 'userid' and 'clean_text' in df.columns:
        button_container_expl = st.container()
        col1, col2 = button_container_expl.columns([0.17, 0.9])
        visualize_button = col1.button("Explore communities")
        hide_button = col2.button("Hide analysis")
        if visualize_button:
            st.session_state.show_second_plot = True
        if hide_button:
            st.session_state.show_second_plot = False

        if st.session_state.show_second_plot:
            with st.spinner('Preparing plot'):
                fig1, fig2, detected_communities, used_texts = explore_communities(
                    df,
                    sample_size,
                    similarity_threshold
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)

            community_options = sorted(set(detected_communities.values()))
            selected_community = st.selectbox('Select a Community', community_options)

            community_users = [user for user, community in detected_communities.items() if
                               community == selected_community]
            community_texts = used_texts[used_texts['userid'].isin(community_users)]['clean_text']

            st.write(f'Post texts from Community {selected_community}:')
            st.markdown('<hr>', unsafe_allow_html=True)
            for text in community_texts:
                st.write(text)

    elif 'clean_text' in df.columns:
        st.error("There is an issue with UserID column. Communities cannot be explored.")
    elif 'userid' in df.columns:
        st.error("There is an issue with Text column. Communities cannot be explored.")
    else:
        st.error("There is an issue with UserID and Text columns. Communities cannot be explored.")

    st.markdown('<hr>', unsafe_allow_html=True)

    st.sidebar.subheader("Additional")

    show_correlation_matrix = st.sidebar.checkbox("Show correlation matrix", False)

    if show_correlation_matrix is True:
        st.subheader("Correlation matrix")
        cols = list(set(available_features) - {'text'})
        corr_df = df[cols]
        with st.spinner('Preparing matrix'):
            corr_matrix = corr_df.corr()
            st.dataframe(corr_matrix, width=800)

else:
    st.info(constants.START_SCREEN)
