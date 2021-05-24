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
from transformers import BertTokenizer

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

    for file in files:
        with open(file) as f:
            data = json.load(f)
        for e in data["episodes"]:
            for s in e["scenes"]:
                for u in s["utterances"]:
                    utterance_ids.append(u['utterance_id'])
                    utterances.append(u['transcript'])

                    if len(u['speakers']) == 0 or len(u['speakers']) > 1:
                        label = speaker_to_label["other"]

                    # Filter side characters
                    elif u['speakers'][0] not in speaker_to_label:
                        label = speaker_to_label["other"]

                    # Keep main characters
                    else:
                        label = speaker_to_label[u['speakers'][0]]

                    labels.append(label)

    tokenized_utterances = tokenizer(utterances, truncation=True, padding=True)

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
