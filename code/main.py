"""
CDM Research project by
- Daniel van de Pavert
-
-
-

something something main description
"""

from dataloader import *
from preprocess import *
from models import *

test_file = 'data/friends_test.json'
train_file = 'data/friends_train.json'
dev_file = 'data/friends_dev.json'


def main(input):
    df = load_data(test_file)
    # get_stats(df)
    df_clean = preprocess(df, ["flags"])

    show_scene(df, 8)


def show_scene(df, scene):

    print("Scene: " + str(scene))
    for line in df.T[scene]:
        if line:
            print(line['speaker'] + ": " + line['utterance'])


if __name__ == '__main__':
    main('PyCharm')
