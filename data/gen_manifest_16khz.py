'''
create train & validation sets for 16khz data with the same filenames as the 24khz data
(the correct file list of gigaspeech xs dataset)
'''


import os
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk


def list_files_in_splits():
    '''
    list all filenames in train, validation & test split of gigaspeech xs
    '''

    # load dataset
    my_dataset_path = '/home/thivt1/data/datasets/my_gigaspeech_dataset'

    # Load the dataset from disk
    new_dataset = load_from_disk(my_dataset_path)

    # save the filenames to txt files for each split
    for split in ['train', 'validation', 'test']:
        print(f'len({split}): {len(new_dataset[split])}')
        with open(os.path.join(my_dataset_path, f'{split}.txt'), 'w') as f:
            for sample in tqdm(new_dataset[split]):
                audio_path = sample['audio']['path']
                file_name = os.path.basename(os.path.splitext(audio_path)[0])
                f.write(f'{file_name}\n')


def create_train_val():
    '''create train.txt and validation.txt for training'''

    # load enc files and phn files
    enc_folder = "/home/thivt1/data/datasets/gigaspeech_xs_phn_enc_manifest/xs/encodec_16khz_4codebooks"
    phn_folder = "/home/thivt1/data/datasets/gigaspeech_xs_phn_enc_manifest/xs/phonemes"

    enc_files = os.listdir(enc_folder)
    phn_files = os.listdir(phn_folder)

    print(len(enc_files))
    print(len(phn_files))

    print(len(set(enc_files) - set(phn_files)))
    print(len(set(phn_files) - set(enc_files)))

    # load train & val filenames
    my_dataset_path = '/home/thivt1/data/datasets/my_gigaspeech_dataset'
    train_set = set()
    val_set = set()
    with open(os.path.join(my_dataset_path, 'train.txt'), 'r') as f:
        for line in f:
            train_set.add(line.strip().replace("train_", ""))
    with open(os.path.join(my_dataset_path, 'validation.txt'), 'r') as f:
        for line in f:
            val_set.add(line.strip().replace("validation_", ""))
    
    print(f'first 5 samples in train_set: {list(train_set)[:5]}')

    # create list of [0, name, code_len] for data sample
    dataset = []  # 0 \t name \t code_len
    for file in tqdm(enc_files):
        name = file.split(".")[0]
        codes = np.loadtxt(os.path.join(enc_folder, file))
        code_len = len(codes[0])
        dataset.append(['0', name, code_len])

    # split dataset into train and validation
    train = []
    validation = []
    for sample in dataset:
        if sample[1] in train_set:
            train.append(sample)
        elif sample[1] in val_set:
            validation.append(sample)

    print('there are {} samples in train'.format(len(train)))
    print('there are {} samples in validation'.format(len(validation)))

    # Define file paths
    train_file_path = '/home/thivt1/data/datasets/gigaspeech_xs_phn_enc_manifest/xs/manifest/train.txt'
    validation_file_path = '/home/thivt1/data/datasets/gigaspeech_xs_phn_enc_manifest/xs/manifest/validation.txt'

    # Write to train.txt
    with open(train_file_path, 'w') as train_file:
        for sample in train:
            train_file.write(f'{sample[0]}\t{sample[1]}\t{sample[2]}\n')

    # Write to validation.txt
    with open(validation_file_path, 'w') as validation_file:
        for sample in validation:
            validation_file.write(f'{sample[0]}\t{sample[1]}\t{sample[2]}\n')


if __name__ == "__main__":

    # list_files_in_splits()
    create_train_val()
