"""
CDM Research project by
- Daniel van de Pavert
-
-
-

something something main description
"""

import pandas
import os

os.chdir("..")


def load_data(file):
    df_test = pandas.read_json(file, orient='records')
    return df_test


def get_stats(df):
    print(df.describe())
