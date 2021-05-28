"""
CDM Research project by
- Daniel van de Pavert
-
-
-

something something preprocess description
"""
import json
import pickle
import numpy as np

from transformers import BertTokenizer
from collections import defaultdict


speaker_to_label = {
    "Chandler Bing": 0,
    "Joey Tribbiani": 1,
    "Monica Geller": 2,
    "Phoebe Buffay": 3,
    "Rachel Green": 4,
    "Ross Geller": 5,
    "other": 6
}

def preprocess():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    files = [
        '../data/json/friends_season_01.json',
        '../data/json/friends_season_02.json',
        '../data/json/friends_season_03.json',
        '../data/json/friends_season_04.json',
        '../data/json/friends_season_05.json',
        '../data/json/friends_season_06.json',
        '../data/json/friends_season_07.json',
        '../data/json/friends_season_08.json',
        '../data/json/friends_season_09.json',
        '../data/json/friends_season_10.json'
    ]

    utterance_ids = []
    utterances = []
    labels = []

    naive_history = True
    all_utterances = False

    for file in files:
        with open(file) as f:
            data = json.load(f)
        for e in data["episodes"]:
            for s in e["scenes"]:
                if naive_history:
                    for u1, u2, u3, u4 in zip(s["utterances"][0:-3], s["utterances"][1:-2], s["utterances"][2:-1], s["utterances"][3:]):
                        if len(u3['speakers']) == 1:
                            # Filter side characters
                            if u3['speakers'][0] in speaker_to_label:
                                label = speaker_to_label[u3['speakers'][0]]

                                utterance_ids.append(u3['utterance_id'])
                                utterances.append(u1['transcript'] + "[SEP]" + u2['transcript'] + "[SEP]" + u3['transcript'] + "[SEP]" + u4['transcript'])
                                labels.append(label)   
                            else:
                                label = speaker_to_label["other"]

                                utterance_ids.append(u3['utterance_id'])
                                utterances.append(u1['transcript'] + "[SEP]" + u2['transcript'] + "[SEP]" + u3['transcript'] + "[SEP]" + u4['transcript'])
                                labels.append(label)                     
                
                else:
                    speaker_dict = defaultdict(list)
                    for u in s["utterances"]:
                        if all_utterances:
                            # Only utterances with one character
                            if len(u['speakers']) == 1:
                                # Filter side characters
                                if u['speakers'][0] in speaker_to_label:
                                    label = speaker_to_label[u['speakers'][0]]

                                    if label not in speaker_dict:
                                        speaker_dict[label] = (u['utterance_id'], u['transcript'])
                                    else:
                                        speaker_dict[label] = (u['utterance_id'], speaker_dict[label][1] + '\n' + u['transcript'])
                                else:
                                    label = speaker_to_label["other"]

                                    if label not in speaker_dict:
                                        speaker_dict[label] = (u['utterance_id'], u['transcript'])
                                    else:
                                        speaker_dict[label] = (u['utterance_id'], speaker_dict[label][1] + '\n' + u['transcript'])

                    # for speaker in speaker_dict:                        
                    #     utterance_ids.append(speaker_dict[speaker][0])
                    #     utterances.append(speaker_dict[speaker][1])
                    #     labels.append(speaker)

                        else:
                            # Only utterances with one character
                            if len(u['speakers']) == 1:
                                # Filter side characters
                                if u['speakers'][0] in speaker_to_label:
                                    label = speaker_to_label[u['speakers'][0]]

                                    utterance_ids.append(u['utterance_id'])
                                    utterances.append(u['transcript'])
                                    labels.append(label)
                                else:
                                    label = speaker_to_label["other"]

                                    utterance_ids.append(u['utterance_id'])
                                    utterances.append(u['transcript'])
                                    labels.append(label)

    tokenized_utterances = tokenizer(utterances, truncation=True, padding=True, add_special_tokens=True)

    scene_id = None
    scene = None
    for id, input_ids, attention_mask, label in zip(
        utterance_ids,
        tokenized_utterances['input_ids'],
        tokenized_utterances['attention_mask'],
        labels
    ):
        if id[:11] != scene_id:
            if scene_id is not None:
                with open('../data/scenes/{}'.format(scene_id), 'wb') as file:
                    pickle.dump(scene, file)
            scene_id = id[:11]
            scene = {}

        scene[id] = (input_ids, attention_mask, label)

    with open('../data/scenes/{}'.format(scene_id), 'wb') as file:
        pickle.dump(scene, file)

    with open('../data/utterance_ids.txt', 'w') as f:
        for id in utterance_ids:
            f.write(id + '\n')

if __name__ == '__main__':
    preprocess()
