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
    s = format_data(df_test)
    return s


def get_stats(df):
    print(df.describe())


def format_data(df):
    something = {}

    for scene in df.T:
        for line in df.T[scene]:
            if line:
                something[line['speaker']] = line['utterance']

    return something
