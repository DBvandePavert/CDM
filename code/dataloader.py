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
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertForPreTraining
from transformers.file_utils import PaddingStrategy

# os.chdir("..")


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


def create_splits(dataset, ratios):
    """
    Args:
        dataset (Dataset): PyTorch dataset object
        ratios (list): List of ratios to split, e.g. [0.8, 0.1, 0.1]
    """
    assert np.sum(ratios) == 1

    # calculate lengths of split according to ratios
    length = len(dataset)
    lengths = [ratio*length for ratio in ratios]
    floored_lengths = [int(np.floor(length)) for length in lengths]
    remainders = [original - floored for original, floored in zip(lengths, floored_lengths)]
    lengths = floored_lengths
    for i in range(length - np.sum(lengths)):
        index = np.argsort(remainders)[::-1][i]
        lengths[index] = lengths[index] + 1

    # returns splits as tuples
    return random_split(dataset, lengths)


class FriendsDataset(Dataset):
    def __init__(self, json_files):
        """
        Args:
            json_file (string): Path to the json file with annotations.
        """

        with open('../data/utterance_ids.txt') as f:
            self.utterance_ids = [line.rstrip('\n') for line in f.readlines()]


    def __len__(self):
        return len(self.utterance_ids)


    def __getitem__(self, idx):
        id = self.utterance_ids[idx]
        scene_id = id[:11]
        with open('../data/scenes/{}'.format(scene_id), 'rb') as f:
            scene = pickle.load(f)

        input_ids, attention_mask, label = scene[id]
        return (
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor([label])
        )


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
    ])
    train_set, val_set, test_set = create_splits(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_set, shuffle=True, num_workers=0)
