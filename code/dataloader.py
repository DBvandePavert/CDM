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
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForPreTraining

os.chdir("..")


class FriendsDataset(Dataset):

    def __init__(self, json_file, tokenizer=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
        """

        with open(json_file) as f:
            data = json.load(f)

        self.foo = []

        for e in data["episodes"]:
            for s in e["scenes"]:

                zap = []

                for u in s["utterances"]:
                    bar = tokenizer(u['transcript'])
                    if len(u['speakers']) == 0:
                        speaker = 'none'
                    else:
                        speaker = u['speakers'][0]
                    zap.append([s["scene_id"], speaker, bar])
                self.foo.append(zap)

    def __len__(self):
        return len(self.foo)

    def __getitem__(self, idx):
        return self.foo[idx]


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = FriendsDataset(json_file='data/json/friends_season_01.json', tokenizer=tokenizer.encode)

dataloader = DataLoader(dataset, shuffle=True, num_workers=0)  # Some weird stuff with the inclusion of batch sizes

for x in dataloader:
    break
