"""
CDM Research project by
- Daniel van de Pavert
-
-
-
TODO something something main description
"""
import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForPreTraining
from transformers.file_utils import PaddingStrategy

os.chdir("..")


speaker_to_label = {
    "Chandler Bing": 0,
    "Joey Tribbiani": 1,
    "Monica Geller": 2,
    "Phoebe Buffay": 3,
    "Rachel Green": 4,
    "Ross Geller": 5,
    "other": 6
}


def create_dataset(json_files):

    ids = []
    utterances = []
    speakers = []

    for file in json_files:
        with open(file) as f:
            data = json.load(f)
        for e in data["episodes"]:
            for s in e["scenes"]:
                for u in s["utterances"]:

                    # Filter utterances with zero or more than one speakers
                    if len(u['speakers']) == 0 or len(u['speakers']) > 1:
                        speaker = speaker_to_label["other"]

                    # Filter side characters
                    elif u['speakers'][0] not in speaker_to_label:
                        speaker = speaker_to_label["other"]

                    # Keep main characters
                    else:
                        speaker = speaker_to_label[u['speakers'][0]]

                    ids.append(s["scene_id"])
                    utterances.append(u['transcript'])
                    speakers.append(speaker)
    return ids, utterances, speakers


class FriendsDataset(Dataset):
    def __init__(self, json_files, tokenizer=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
        """
        # Tokenize utterances
        ids, utterances, speakers = create_dataset(json_files)
        tokenized_utterances = tokenizer(utterances, truncation=True, padding=True)

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


    def speaker_to_label(self, str):
        return {
            "Chandler Bing": 0,
            "Joey Tribbiani": 1,
            "Monica Geller": 2,
            "Phoebe Buffay": 3,
            "Rachel Green": 4,
            "Ross Geller": 5,
            "other": 6
        }[str]


    def label_to_speaker(self, str):
        return {
            0: "Chandler Bing",
            1: "Joey Tribbiani",
            2: "Monica Geller",
            3: "Phoebe Buffay",
            4: "Rachel Green",
            5: "Ross Geller",
            6: "other"
        }[str]

    
    def num_labels(self):
        return 7


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = FriendsDataset([
        'data/json/friends_season_01.json',
        'data/json/friends_season_02.json',
        'data/json/friends_season_03.json',
        'data/json/friends_season_04.json',
        'data/json/friends_season_05.json',
        'data/json/friends_season_06.json',
        'data/json/friends_season_07.json',
        'data/json/friends_season_08.json',
        'data/json/friends_season_09.json',
        'data/json/friends_season_10.json'
    ], tokenizer=tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, num_workers=0)
