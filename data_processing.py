import pandas as pd
import streamlit as st

import constants
import text_processing


def validate_dataframe(dataframe):
    columns = dataframe.columns
    missing_columns = set(constants.TARGETS) - set(columns)
    violations = []
    constraints = []
    if len(missing_columns) == 2:
        violations.append("Dataframe is missing target columns")
    if "text" not in columns:
        constraints.append("No text column - visualized data won't include labels for uploaded posts")
    if "userid" not in columns:
        constraints.append("No column containing user info - impossible to explore communities")
    if "language" not in columns and "lang" not in columns:
        constraints.append("No language column - visualized data might be incorrect")
    else:
        constraints.append("Language column detected - only records in English will be kept")
    return constraints, violations


def clean_text(dataframe):
    cleaned_df = dataframe.copy()
    if 'language' in cleaned_df.columns:
        cleaned_df = cleaned_df[
            cleaned_df['language'].isin(['en', 'En', 'EN', 'eng', 'Eng', 'ENG', 'english', 'English', 'ENGLISH'])]
    if 'lang' in cleaned_df.columns:
        cleaned_df = cleaned_df[
            cleaned_df['lang'].isin(['en', 'En', 'EN', 'eng', 'Eng', 'ENG', 'english', 'English', 'ENGLISH'])]
    cleaned_df["clean_text"] = cleaned_df['text'].apply(text_processing.clean)
    cleaned_df['clean_text'] = cleaned_df['clean_text'].apply(text_processing.remove_stopwords)
    return cleaned_df


def map_sentiment(dataframe):
    map_df = dataframe.copy()
    sentiment_mapping = {
        'negative': -1,
        'neutral': 0,
        'positive': 1
    }
    map_df['sentiment'] = (
        map_df['sentiment'].apply(lambda x: sentiment_mapping.get(x.strip().lower(), 0) if isinstance(x, str) else x)
    )
    return map_df


def get_data(datafiles):
    if datafiles is None or len(datafiles) == 0:
        return None

    first_filename = datafiles[0]
    if first_filename.name.endswith('.csv'):
        df_raw = pd.read_csv(first_filename)
    elif first_filename.name.endswith('.xlsx'):
        df_raw = pd.read_excel(first_filename)
    else:
        return None
    df_raw.columns = [x.lower() for x in df_raw.columns]
    columns = df_raw.columns

    for idx in range(1, len(datafiles)):
        file = datafiles[idx]
        if file.name.endswith('.csv'):
            df_file = pd.read_csv(file)
            if list(columns) != list(df_file.columns):
                st.sidebar.error("Files must contain data with the same columns.")
                return None
            else:
                df_raw = pd.concat([df_raw, df_file], ignore_index=True)
        elif file.name.endswith('.xlsx'):
            df_file = pd.read_excel(file)
            if list(columns) != list(df_file.columns):
                st.sidebar.error("Files must contain data with the same columns.")
                return None
            else:
                df_raw = pd.concat([df_raw, df_file], ignore_index=True)
    constraints, violations = validate_dataframe(df_raw)
    if violations:
        st.sidebar.error("**Violations:**\n" + "\n".join([f"- {violation}" for violation in violations]))
        return None
    if constraints:
        st.sidebar.warning("**Warnings:**\n" + "\n".join([f"- {constraint}" for constraint in constraints]))
    if 'text' in columns:
        df_raw = clean_text(df_raw)
    if 'sentiment' in columns:
        df_raw = map_sentiment(df_raw)
    return df_raw


def get_available_features(dataframe):
    available_features = [x for x in dataframe.columns if x in constants.ENGAGEMENT_FEATURES]
    if 'text' in dataframe.columns:
        available_features.append("text")
    return available_features


def get_available_targets(dataframe):
    available_targets = [x for x in dataframe.columns if x in constants.TARGETS]
    return available_targets
