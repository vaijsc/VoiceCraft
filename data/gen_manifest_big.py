'''
generate manifest (train.txt and validation.txt) for big dataset
(combine vin27 4k hours and sach noi 1k hours)
'''

import json
from tqdm.contrib.concurrent import thread_map
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_code_len(segment_id):
    codes = np.loadtxt(os.path.join(enc_folder, segment_id + '.txt'))
    code_len = codes.shape[1]
    return ['0', segment_id, code_len]


if __name__ == '__main__': 
    # load json file
    # this file include: train set of vin27 + sach noi, validation set of vin27, test set of vin27 + sach noi
    dataset_path = '/workspace/VoiceCraft/data/combined_data.json'
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    print(f'first 5 samples in dataset: {dataset[:5]}\n')

    # load validation and test set of vin27
    with open("/workspace/oneshot/linh_transfer/vin27_dev.jsonl", 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        vin27_validation = set([item['path'].replace("vin27", "vin27_16k") for item in data])
    print(f'first 5 samples in vin27 validation: {list(vin27_validation)[:5]}\n')

    with open("/workspace/oneshot/linh_transfer/vin27_test.jsonl", 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        vin27_test = set([item['path'].replace("vin27", "vin27_16k") for item in data])
    print(f'first 5 samples in vin27 test: {list(vin27_test)[:5]}\n')

    # load train & test of sach noi
    train_sach_noi_path = '/workspace/VoiceCraft/data/sach_noi_train.json'
    test_sach_noi_path = '/workspace/VoiceCraft/data/sach_noi_test.json'
    with open(train_sach_noi_path, 'r', encoding='utf-8') as file:
        sach_noi_train = set([item['path'] for item in json.load(file)])
    print(f'first 5 samples in sach noi train: {list(sach_noi_train)[:5]}\n')

    with open(test_sach_noi_path, 'r', encoding='utf-8') as file:
        sach_noi_test = set([item['path'] for item in json.load(file)])
    print(f'first 5 samples in sach noi test: {list(sach_noi_test)[:5]}\n')

    # segment_id of train & validation set
    train_segment_list = []
    validation_segment_list = []
    count_vin27_train = 0
    count_vin27_valid = 0
    count_vin27_test = 0
    count_sn_train = 0
    count_sn_test = 0
    for item in dataset: 
        path = item['path']
        if 'chunk' in path: # data sach noi 
            if path in sach_noi_train: 
                train_segment_list.append(item['segment_id'])
                count_sn_train += 1
            elif path in sach_noi_test:
                count_sn_test +=1
            else: 
                raise ValueError(f'path {path} not in sach noi train or test')
        else: # data vin27
            if path in vin27_validation:
                if "/lustre/scratch/client/vinai/users/linhnt140/zero-shot-tts/preprocess_audio/vin27_16k/đa-nang/0327369/086.wav" == path: 
                    print('hehe')
                validation_segment_list.append(item['segment_id'])
                count_vin27_valid += 1 
            elif path in vin27_test:
                count_vin27_test += 1
            else: 
                train_segment_list.append(item['segment_id'])
                count_vin27_train += 1 
            
    print('there are {} samples in train'.format(len(train_segment_list)))
    print('there are {} samples in validation'.format(len(validation_segment_list)))
    print(f'len(dataset): {len(dataset)}')
    print(f'count_vin27_train: {count_vin27_train}')
    print(f'count_vin27_valid: {count_vin27_valid}')
    print(f'count_sn_train: {count_sn_train}')
    print(f'count_vin27_test: {count_vin27_test}')
    print(f'count_sn_test: {count_sn_test}')
        
    # use multithread to get code_len of each segment_id
    train = []
    validation = []
    max_workers = 8 
    enc_folder = "/home/thivt1/data/datasets/big/phn_codes/encodec_16khz_4codebooks"

    # get code_len of validation data
    validation = thread_map(get_code_len, validation_segment_list, max_workers=max_workers, desc='getting code len of validation data')
    print(f'len(validation) results: {len(validation)}')

    # get code_len of training data
    train = thread_map(get_code_len, train_segment_list, max_workers=max_workers, desc='getting code len of train data')
    print(f'len(train) results: {len(train)}')

    # Define file paths
    base_dir = os.path.dirname(enc_folder)
    train_save_path = os.path.join(base_dir, 'manifest/train.txt')
    validation_save_path = os.path.join(base_dir, 'manifest/validation.txt')

    # Write to train.txt
    os.makedirs(os.path.dirname(train_save_path), exist_ok=True)
    with open(train_save_path, 'w') as train_file:
        for sample in train:
            train_file.write(f'{sample[0]}\t{sample[1]}\t{sample[2]}\n')

    # Write to validation.txt
    with open(validation_save_path, 'w') as validation_file:
        for sample in validation:
            validation_file.write(f'{sample[0]}\t{sample[1]}\t{sample[2]}\n')

 
            
    
    
