'''
generate manifest (train.txt and validation.txt) for sachnoi dataset 
(940 hours + 500 hours augmented = 1500 hours)
'''

import hashlib
import pandas as pd
from tqdm.contrib.concurrent import thread_map
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_segment_id_from_path(filepath): 
    '''
    input: 
        path: 'big_processed_data/Khương_Ngọc_Đình/Hồi_Ký_Của_Các_Tướng_Tá_Sài_Gòn/tuong-ta-3/chunk-4423_22-4426_94.wav'
    output: 
        segment_id from path
    '''
    # return "-".join(path.split('/')[3:]).replace('.wav', '').strip()
    return hashlib.md5(filepath.encode()).hexdigest()

def process_sample(enc_folder, file):
    codes = np.loadtxt(os.path.join(enc_folder, file))
    code_len = codes.shape[1]
    name = os.path.splitext(file)[0] # splitext can handle cases where file has . in its name
    # name = get_segment_id_from_path(file)
    return ['0', name, code_len]

def check_train_val(sample, train_set, val_set):
    # there might be overlap between train & val segment_ids, 
    # in which case we prioritize val_set
    if sample[1] in val_set:
        return 'validation', sample
    elif sample[1] in train_set:
        return 'train', sample
    else:
        return None, sample

if __name__ == '__main__': 
    # read train data for sach noi
    train_path = '/lustre/scratch/client/vinai/users/thivt1/code/zstts/split_long_audio/metadata/step20_train.csv'
    train_data = pd.read_csv(train_path, sep='|', header=None, names=['audio_path', 'transcript', 'speaker', 'duration'])
    train_data = train_data.values.tolist()
    # extract the path 
    train_audio_paths = [item[0] for item in train_data]
    # extract the segment_id
    train_segment_ids = [get_segment_id_from_path(path) for path in train_audio_paths]
    print(f'first 5 train segment_ids: {train_segment_ids[:5]}')
    train_set = set(train_segment_ids)

    # read val data
    val_path = '/lustre/scratch/client/vinai/users/thivt1/code/zstts/split_long_audio/metadata/step20_val.csv'
    val_data = pd.read_csv(val_path, sep='|', header=None, names=['audio_path', 'transcript', 'speaker', 'duration'])
    val_data = val_data.values.tolist()
    val_audio_paths = [item[0] for item in val_data]
    val_segment_ids = [get_segment_id_from_path(path) for path in val_audio_paths]
    print(f'first 5 val segment_ids: {val_segment_ids[:5]}')
    val_set = set(val_segment_ids)

    # read phone folder
    meta_folder = '/home/thivt1/data/datasets/sachnoi_1500hr'
    phn_folder = os.path.join(meta_folder, 'phonemes')
    enc_folder = os.path.join(meta_folder, 'encodec_16khz_4codebooks')
    enc_files = os.listdir(enc_folder)
    phn_files = os.listdir(phn_folder)
    print(f'len(enc_files): {len(enc_files)}')
    print(f'len(phn_files): {len(phn_files)}')

    # Create list of [0, name, code_len] for data sample using multithreading
    dataset = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_sample, enc_folder, file) for file in enc_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing enc files'):
            dataset.append(future.result())

    # load dataset from file
    # dataset_path = os.path.join(meta_folder, 'dataset.csv')
    # dataset = pd.read_csv(dataset_path).values.tolist()
    # enc_files_set = [item[1] for item in dataset]

    # save dataset to csv file
    dataset_path = os.path.join(meta_folder, 'dataset.csv')
    dataset_df = pd.DataFrame(dataset, columns=['0', 'name', 'code_len'])
    dataset_df.to_csv(dataset_path, index=False)

    # Split dataset into train and validation using multithreading
    train = []
    validation = []
    none_list = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_train_val, sample, train_set, val_set) for sample in dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Splitting dataset'):
            category, sample = future.result()
            if category == 'train':
                train.append(sample)
            elif category == 'validation':
                validation.append(sample)
            else: 
                none_list.append(sample)

    print('there are {} samples in train'.format(len(train)))
    print('there are {} samples in validation'.format(len(validation)))

    # Define file paths
    train_file_path = os.path.join(meta_folder, 'manifest/train.txt')
    validation_file_path = os.path.join(meta_folder, 'manifest/validation.txt')

    # Write to train.txt
    os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
    with open(train_file_path, 'w') as train_file:
        for sample in train:
            train_file.write(f'{sample[0]}\t{sample[1]}\t{sample[2]}\n')

    # Write to validation.txt
    with open(validation_file_path, 'w') as validation_file:
        for sample in validation:
            validation_file.write(f'{sample[0]}\t{sample[1]}\t{sample[2]}\n')
    