"""
CDM Research project by
- Daniel van de Pavert
-
-
-
TODO something something main description
"""
import pandas
import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForPreTraining
from transformers.file_utils import PaddingStrategy

os.chdir("..")


speaker_to_class = {
    "Chandler Bing": 0,
    "Joey Tribbiani": 1,
    "Monica Geller": 2,
    "Phoebe Buffay": 3,
    "Rachel Green": 4,
    "Ross Geller": 5,
    "other": 6
}


def create_dataset(json_file):
    with open(json_file) as f:
        data = json.load(f)

    ids = []
    utterances = []
    speakers = []

    for e in data["episodes"]:
        for s in e["scenes"]:
            for u in s["utterances"]:
                utterance = u['transcript']
                
                # Filter utterances with zero or more than one speakers
                if len(u['speakers']) == 0 or len(u['speakers']) > 1:
                    speaker = speaker_to_class["other"]

                # Filter side characters
                elif u['speakers'][0] not in speaker_to_class:
                    speaker = speaker_to_class["other"]

                # Keep main characters
                else:
                    speaker = speaker_to_class[u['speakers'][0]]

                ids.append(s["scene_id"])
                utterances.append(u['transcript'])
                speakers.append(speaker)
    return ids, utterances, speakers


class FriendsDataset(Dataset):
    def __init__(self, json_file, tokenizer=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
        """
        # Tokenize utterances
        ids, utterances, speakers = create_dataset(json_file)
        tokenized_utterances = tokenizer(utterances[:2], truncation=True, padding=True)

        # Construct a tokenized padded dataset list
        self.dataset = []
        for id, utterance, speaker in zip(tokenized_utterances["input_ids"], tokenized_utterances["attention_mask"], speakers):
            self.dataset.append((
                torch.tensor(id),
                torch.tensor(utterance),
                torch.tensor([speaker])
            ))


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = FriendsDataset(json_file='data/json/friends_season_01.json', tokenizer=tokenizer)
dataloader = DataLoader(dataset, shuffle=True, num_workers=0)
