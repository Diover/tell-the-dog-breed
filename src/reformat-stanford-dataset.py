import numpy as np
import pandas as pd
import os
import re
from os.path import join
from os import path, makedirs, rename
from tqdm import tqdm


def get_new_name(folder_name, labels):
    folder_name = folder_name.lower()
    breed_name_regex = re.compile("[n\d]+-(?P<breed_name>[\w_]+)")
    breed_name = breed_name_regex.match(folder_name).group("breed_name")
    return breed_name if breed_name in labels else None


def save_data_to_folders(input_folder):
    labels_csv = pd.read_csv(join(input_folder, "labels.csv"))
    breeds = pd.Series(labels_csv['breed'])
    labels = np.unique(breeds)
    labels.sort()

    image_base_folder = "{}/images".format(input_folder)
    image_folders = [file for file in os.listdir(image_base_folder) if
                     os.path.isdir(os.path.join(image_base_folder, file))]
    
    image_folders_rename_map = {(folder_name, get_new_name(folder_name, labels)) for folder_name in image_folders}

    for old_name, new_name in tqdm(image_folders_rename_map):
        if new_name is None:
            continue
        os.rename(os.path.join(image_base_folder, old_name), os.path.join(image_base_folder, new_name))
    # unique_breeds = np.unique(breeds)
    # labels = []
    # for breed in breeds:
    #     i = np.where(unique_breeds == breed)[0][0]
    #     labels.append(i)
    #
    # n_breeds = unique_breeds.size
    # labels = np.eye(n_breeds)[labels]
    #
    # filenames_train = []
    # filenames_validate = []
    #
    # # move to validate folder
    # for i in tqdm(range(len(filenames))):
    #     label = unique_breeds[np.where(labels[i] == 1.)][0]
    #     filename = '{}.jpg'.format(filenames[i])
    #
    #     if i < 8000:
    #         new_dir = input_folder + '/sorted/train/{}/'.format(label)
    #         filenames_train.append(new_dir + filename)
    #     else:
    #         new_dir = input_folder + '/sorted/validate/{}/'.format(label)
    #         filenames_validate.append(new_dir + filename)
    #
    #     if not path.exists(new_dir):
    #         makedirs(new_dir)
    #
    #     rename(input_folder + "/train/{}.jpg".format(filenames[i]), new_dir + filename)

#save_data_to_folders("../input")
