import pandas as pd
import streamlit as st

import constants
import text_processing


def validate_dataframe(dataframe):
    columns = [x.lower() for x in dataframe.columns]
    missing_columns = set(constants.TARGETS) - set(columns)
    violations = []
    constraints = []
    columns = [x.lower() for x in dataframe.columns]
    if len(missing_columns) == 2:
        violations.append("Dataframe is missing label columns")
    if "lang" not in columns:
        constraints.append("No language column - visualized data might be incorrect")

    return constraints, violations


def clean_text(dataframe):
    cleaned_df = dataframe.copy()
    if "lang" in cleaned_df.columns:
        cleaned_df = cleaned_df[(cleaned_df["lang"] == 'en') | (cleaned_df["lang"] == 'english')]
    if "Lang" in cleaned_df.columns:
        cleaned_df = cleaned_df[(cleaned_df["Lang"] == 'en') | (cleaned_df["Lang"] == 'english')]
    cleaned_df["clean_text"] = cleaned_df["text"].apply(text_processing.clean)
    cleaned_df['clean_text'] = cleaned_df['clean_text'].apply(text_processing.remove_stopwords)
    return cleaned_df


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
    if "text" in columns:
        df_raw = clean_text(df_raw)
    return df_raw


def get_available_features(dataframe):
    columns = {}
    available_features = []
    available_columns = []
    for x in dataframe.columns:
        columns[x.lower()] = x
    for key, value in columns.items():
        if key in constants.ENGAGEMENT_FEATURES:
            available_features.append(key)
            available_columns.append(value)
    if 'text' in columns.keys():
        available_features.append("text")
        available_columns.append(columns["text"])
    return available_features, available_columns


def get_available_targets(dataframe):
    columns = [x.lower() for x in dataframe.columns]
    available_targets = []
    if 'topic' in columns:
        available_targets.append('Topic')
    if 'sentiment' in columns:
        available_targets.append('Sentiment')
    return available_targets
