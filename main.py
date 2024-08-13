import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import networkx as nx
import matplotlib.pyplot as plt
import base64
import csv

st.set_page_config(layout='wide')

import matplotlib.patches as mpatches

def create_network_graph(df: pd.DataFrame, source, destination, node_color='blue', edge_color='black', node_size=100) -> None:
    """
    Creates a directed network graph from two columns and displays it on Streamlit app

    Args:
        df (pd.DataFrame): Original dataframe
        source (str): Name of the column to use as the source node
        destination (str): Name of the column to use as the destination node
        node_color (str): Color of the nodes (default: 'blue')
        edge_color (str): Color of the edges (default: 'black')
        node_size (int): Size of the nodes (default: 100)
    """

    # Create a directed graph from two columns
    G = nx.from_pandas_edgelist(df, source=str(source), target=str(destination))

    # Add Net Effect column to each edge in the graph
    # ...

    # Draw the network graph using Matplotlib and show it on Streamlit app
    fig, ax = plt.subplots(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.25, scale=10)  # positions for all nodes
    nx.draw_networkx(G, pos, with_labels=True, ax=ax, node_color=node_color, edge_color=edge_color, node_size=node_size)
    labels = nx.get_edge_attributes(G, 'Net Effect')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Create legend
    legend_patches = [
        mpatches.Patch(color=node_color, label=str(source)),
        mpatches.Patch(color=edge_color, label=str(destination)),
    ]
    plt.legend(handles=legend_patches)

    st.pyplot(fig)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe


  """

    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_text_input)]

    return df


def check_file_type(file):
    filename = file.name
    file_extension = filename.split(".")[-1].lower()

    if file_extension == 'xlsx':
        return 'xlsx'
    elif file_extension == 'csv':
        return 'csv'
    else:
        return None

def get_csv_delimiter(file):
    content = file.getvalue().decode()
    dialect = csv.Sniffer().sniff(content)
    return dialect.delimiter


@st.cache_resource  # Caching the function for better performance
def load_data(file):
    file_type = file.name.split(".")[-1].lower()
    if file_type is not None:
        if str(file_type) == "xlsx":
            df = pd.read_excel(file, sheet_name='C.1 Employee_Indirect_Creditors')
            return df
        elif str(file_type) == "csv":
            sep = get_csv_delimiter(file)
            df = pd.read_csv(file, sep=sep)
            return df
    else:
        st.write("Cannot find extension!")

def get_source_destination(df):
    source = st.selectbox("Please select the source node:", options=df.columns.tolist(), key='selectbox1')
    destination = st.selectbox("Please select the source node:", options=df.columns.tolist(), key='selectbox2')

    return str(source), str(destination)

def main():
    st.title("Entity Relationships Graph Analysis")

    file = st.file_uploader("Upload Excel File", type=["xlsx", "csv"])

    if file is not None:

        df = load_data(file)

        st.write(df.head())

        source, destination = get_source_destination(df)

        filtered_df = filter_dataframe(df)

        # Show the filtered dataframe on Streamlit app
        st.text("Filtered Data")
        st.dataframe(filtered_df)

        # Additional input fields for style options
        node_color = st.color_picker("Node Color", value="#00FFAA")
        edge_color = st.color_picker("Edge Color", value="#00FFBB")
        node_size = st.slider("Node Size", min_value=1, max_value=100, value=50, step=1)

        # Create and display network graph based on filtered DataFrame with the specified style options
        create_network_graph(filtered_df, source, destination, node_color=node_color, edge_color=edge_color, node_size=node_size)


if __name__ == '__main__':
    main()