"""
CDM Research project by
- Daniel van de Pavert
-
-
-

something something preprocess description
"""

import pandas
from transformers import BertTokenizer, BertForPreTraining

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess(df, flags):

    for key in df.keys():
        df[key] = tokenizer(df[key])

    return df

