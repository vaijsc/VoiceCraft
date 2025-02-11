'''
create new dataset from gigaspeech xs dataset, with sampling rate 24khz
'''

import os
import numpy as np
import shutil
from datasets import load_dataset, DatasetDict, Dataset, DownloadConfig
from tqdm import tqdm

# Load dataset
download_to = "/home/thivt1/data/datasets/gigaspeech_xs"
dc = DownloadConfig(cache_dir=download_to)
dataset = load_dataset("speechcolab/gigaspeech", "xs",
                       token=True, cache_dir=download_to, download_config=dc)

# Specify the output directory where you want to save the audio files
output_dir = '/home/thivt1/data/datasets/my_gigaspeech_dataset/audio_files'
os.makedirs(output_dir, exist_ok=True)

# function to copy audio files from a dataset split to the output directory and update the paths
def copy_audio_files_and_update_paths(dataset_split, split_name, output_dir):
    new_dataset_split = []
    total_duration = 0.0

    for i, sample in tqdm(enumerate(dataset_split), total=len(dataset_split), desc=f"Processing {split_name}"):
        audio_path = sample['audio']['path']
        file_name = os.path.basename(os.path.splitext(audio_path)[0])
        file_extension = os.path.splitext(audio_path)[1]
        new_path = os.path.join(
            output_dir, f'{split_name}_{file_name}{file_extension}')

        # Copy the audio file
        shutil.copy(audio_path, new_path)

        # Update the sample with the new path
        new_sample = sample.copy()
        new_sample['audio']['path'] = new_path
        new_sample['audio']['sampling_rate'] = 24000

        # Calculate the duration using begin_time and end_time
        begin_time = float(sample['begin_time'])
        end_time = float(sample['end_time'])
        duration = end_time - begin_time
        total_duration += duration

    return new_dataset_split, total_duration


def create_new_dataset():
    # Create a new dataset dictionary to hold the new splits
    new_dataset_dict = {}
    total_durations = {'train': 0.0, 'validation': 0.0, 'test': 0.0}

    # Iterate through each split in the dataset, copy the files, and create the new dataset splits
    splits = ['validation']
    for split_name in splits:
        dataset_split = dataset[split_name]
        # Include only validation and test splits in the new dataset
        print(
            f"Processing {split_name} split with {len(dataset_split)} samples")
        new_dataset_split, split_duration = copy_audio_files_and_update_paths(
            dataset_split, split_name, output_dir)
        total_durations[split_name] = split_duration
        new_dataset_dict[split_name] = Dataset.from_list(new_dataset_split)

        print(
            f"Finished processing {split_name} split with total duration: {split_duration} seconds")

    # Create a DatasetDict from the new dataset splits
    new_dataset = DatasetDict(new_dataset_dict)

    # Save the new dataset to disk
    new_dataset_path = '/home/thivt1/data/datasets/my_gigaspeech_dataset'
    os.makedirs(new_dataset_path, exist_ok=True)
    new_dataset.save_to_disk(new_dataset_path)

    # Print total durations
    print("Total durations (seconds):")
    for split_name, duration in total_durations.items():
        print(f"{split_name}: {duration}")

    print("All audio files have been copied and the new dataset has been saved.")


def test_new_dataset():
    from datasets import load_from_disk

    # Specify the path where the new dataset is saved
    new_dataset_path = '/home/thivt1/data/datasets/my_gigaspeech_dataset'

    # Load the dataset from disk
    new_dataset = load_from_disk(new_dataset_path)

    # Print the structure of the dataset to verify successful loading
    print(new_dataset)

    # Optionally, inspect the first few examples in each split
    # for sample in new_dataset['train']:
    #     if "<SIL>" in sample['text'] or "<MUSIC>" in sample['text'] or "<NOISE>" in sample['text'] or "<OTHER>" in sample['text']:
    #         print(sample)
    #         break
    print(new_dataset['train'][0]['segment_id'])
    for sample in tqdm(new_dataset['train'], total=len(new_dataset['train'])):
        if sample['segment_id'] == "YOU1000000044_S0000166": 
            print(sample)

    for sample in tqdm(new_dataset['train'], total=len(new_dataset['validation'])):
        if sample['segment_id'] == "YOU1000000044_S0000166": 
            print(sample)


def find_sample_with_sil():
    import glob

    folder_path = '/home/thivt1/data/datasets/gigaspeech_phn_enc_manifest_8cb/xs/phonemes'
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    for txt_file in txt_files:
        # Open and read the content of the file
        with open(txt_file, 'r', encoding='utf-8') as file:
            content = file.read()
            # Check if '<SIL>' is in the content
            if '<SIL>' in content:
                print(f"The file '{txt_file}' contains '<SIL>'.")
                break
    else:
        print("No file contains special tokens.")


def test_gigaspeech_xs():
    dataset = load_dataset("speechcolab/gigaspeech", "xs", token=True, cache_dir="/home/thivt1/data/datasets/gigaspeech_xs", download_config=dc) 
    sample = dataset['validation'][1]
    print(sample)
    print(sample['audio']['array'].dtype)
    print(sample['audio']['path'])

if __name__ == '__main__':
    # create_new_dataset()
    # test_new_dataset()
    test_gigaspeech_xs()
    # find_sample_with_sil()
