import pickle

def get_number_of_speakers(scene_id):
    with open('data/scenes/{}'.format(scene_id), 'rb') as file:
        scene = pickle.load(file)

    return len(set([speaker for _, _, speaker in scene.values()]))

if __name__ == '__main__':
    print(get_number_of_speakers('s01_e01_c01'))
