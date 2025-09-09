# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:55:12 2025

These helper functions will help with handling demographic data and filtering
for the streamlit app.

@author: kevjm
"""

import pandas as pd
import streamlit as st

def generate_labels(bin_edges, suffix="+"):
    """
    Generate labels like '20–29', '<20', '70+' from bin edges.
    Validates and sorts bin_edges if needed.
    """
    # Validate bin_edges
    if not all(isinstance(edge, (int, float)) for edge in bin_edges):
        raise ValueError("All bin edges must be numeric.")

    # Sort if not ascending
    if bin_edges != sorted(bin_edges):
        bin_edges = sorted(bin_edges)

    # Generate labels
    labels = []
    for ii in range(len(bin_edges) - 1):
        left = bin_edges[ii]
        right = bin_edges[ii + 1]

        if right == float("inf"):
            labels.append(f"{int(left)}{suffix}")
        else:
            labels.append(f"{int(left)}–{int(right - 1)}")

    return labels




#Demographic bins and labels
def bin_demographics(df,col_name,bin_edges,lbls):
    """
    this function bins given demographic features into different groupings.
    inputs:
        df: dataframe with demographic informaiton
        col_name: column name of demographic feature that will be discretized
        bin_edges: list of values to be used as bin edges for demographic values
        lbls: descriptive labels for groupings
    returns:
        A series where each row represents a row in the original data frame
        with the appropriate label
    """
    return pd.cut(df[col_name],
                  bins = bin_edges,
                  labels = lbls)

    # visualize data across column features
def apply_filter(df, column_name, sidebar_label):
    """
    Adds a selectbox to the sidebar with an 'All' option and filters the DataFrame accordingly.
    """
    filter_key = f"{column_name}_filter"
    col_data = df[column_name].dropna()
    
    # If column is categorical, preserve its order
    if pd.api.types.is_categorical_dtype(col_data):
        options = ["All"] + list(col_data.cat.categories)
    else:
        options = ["All"] + sorted(map(str, col_data.unique().tolist()))
    
    selected_value = st.sidebar.selectbox(
        sidebar_label,
        options,
        index=options.index(st.session_state.get(filter_key,"All")),
        key = filter_key
        )

    if selected_value != "All":
        df = df[df[column_name] == selected_value]

    return df, selected_value
