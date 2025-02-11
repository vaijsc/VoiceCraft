'''
generate train.txt and validation.txt for different dataset
'''

import os
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_sample(enc_folder, file):
    name = file.split(".")[0]
    codes = np.loadtxt(os.path.join(enc_folder, file))
    code_len = codes.shape[1]
    return ['0', name, code_len]


def check_train_val(sample, train_set, val_set):
    if sample[1] in train_set:
        return 'train', sample
    elif sample[1] in val_set:
        return 'validation', sample
    else:
        return None, None


def create_train_val(enc_folder, phn_folder, my_dataset_path):
    '''create train.txt and validation.txt for training'''

    # Load enc files and phn files
    enc_files = os.listdir(enc_folder)
    phn_files = os.listdir(phn_folder)

    print(len(enc_files))
    print(len(phn_files))

    print(len(set(enc_files) - set(phn_files)))
    print(len(set(phn_files) - set(enc_files)))

    # Load train & val filenames
    my_dataset = load_from_disk(my_dataset_path)
    train_set = set()
    val_set = set()
    for sample in tqdm(my_dataset['train'], total=len(my_dataset['train']), desc='listing train split'):
        train_set.add(sample['segment_id'])

    for sample in tqdm(my_dataset['validation'], total=len(my_dataset['validation']), desc='listing validation split'):
        val_set.add(sample['segment_id'])

    print(f'first 5 samples in train_set: {list(train_set)[:5]}')

    # Create list of [0, name, code_len] for data sample using multithreading
    dataset = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sample, enc_folder, file) for file in enc_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing enc files'):
            dataset.append(future.result())

    # Split dataset into train and validation using multithreading
    train = []
    validation = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_train_val, sample, train_set, val_set) for sample in dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Splitting dataset'):
            category, sample = future.result()
            if category == 'train':
                train.append(sample)
            elif category == 'validation':
                validation.append(sample)

    print('there are {} samples in train'.format(len(train)))
    print('there are {} samples in validation'.format(len(validation)))

    # Define file paths
    base_dir = os.path.dirname(enc_folder)
    train_file_path = os.path.join(base_dir, 'manifest/train.txt')
    validation_file_path = os.path.join(base_dir, 'manifest/validation.txt')

    # Write to train.txt
    os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
    with open(train_file_path, 'w') as train_file:
        for sample in train:
            train_file.write(f'{sample[0]}\t{sample[1]}\t{sample[2]}\n')

    # Write to validation.txt
    with open(validation_file_path, 'w') as validation_file:
        for sample in validation:
            validation_file.write(f'{sample[0]}\t{sample[1]}\t{sample[2]}\n')


if __name__ == "__main__":
    enc_folder = "/home/thivt1/data/datasets/sach_noi/phn_codes_combined/encodec_16khz_4codebooks"
    phn_folder = "/home/thivt1/data/datasets/sach_noi/phn_codes_combined/phonemes"
    my_dataset_path = '/home/thivt1/data/datasets/sach_noi/combined_dataset'

    create_train_val(enc_folder=enc_folder, phn_folder=phn_folder, my_dataset_path=my_dataset_path)