import numpy as np
import pandas as pd
from os.path import join
from os import path, makedirs, rename
from tqdm import tqdm


def save_data_to_folders(input_folder):
    labels_csv = pd.read_csv(join(input_folder, "labels.csv"))
    breeds = pd.Series(labels_csv['breed'])
    filenames = pd.Series(labels_csv['id'])

    unique_breeds = np.unique(breeds)
    labels = []
    for breed in breeds:
        i = np.where(unique_breeds == breed)[0][0]
        labels.append(i)

    n_breeds = unique_breeds.size
    labels = np.eye(n_breeds)[labels]

    filenames_train = []
    filenames_validate = []

    # move to validate folder
    for i in tqdm(range(len(filenames))):
        label = unique_breeds[np.where(labels[i] == 1.)][0]
        filename = '{}.jpg'.format(filenames[i])

        if i < 8000:
            new_dir = input_folder + '/sorted/train/{}/'.format(label)
            filenames_train.append(new_dir + filename)
        else:
            new_dir = input_folder + '/sorted/validate/{}/'.format(label)
            filenames_validate.append(new_dir + filename)

        if not path.exists(new_dir):
            makedirs(new_dir)

        rename(input_folder + "/train/{}.jpg".format(filenames[i]), new_dir + filename)
