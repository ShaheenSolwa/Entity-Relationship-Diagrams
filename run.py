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


import subprocess

if __name__ == "__main__":
    subprocess.run(r"streamlit run C:\Users\ssolwa001\PycharmProjects\EntityRelationships\main.py")