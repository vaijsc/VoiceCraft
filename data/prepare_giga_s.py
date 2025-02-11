from tqdm import tqdm
from datasets import load_dataset, DownloadConfig
import csv
import pandas as pd


def download(dataset_size, download_to):
    dc = DownloadConfig(cache_dir=download_to)
    dataset = load_dataset("speechcolab/gigaspeech", dataset_size,
                           token=True, cache_dir=download_to, download_config=dc)
    print(dataset)


def dataset2csv(dataset_size, download_to):
    '''
    convert gigaspeech dataset to csv file with neccessary columns:
    segment_id, path, transcript, speaker, duration 
    '''
    dc = DownloadConfig(cache_dir=download_to)
    dataset = load_dataset("speechcolab/gigaspeech", dataset_size,
                           token=True, cache_dir=download_to, download_config=dc,
                           split='train')
        
    # Define the output CSV file path
    output_csv = "/home/thivt1/data/datasets/my_giga_s/full_gigaspeech_s.csv"
    
    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
        
        # Write the header row
        csv_writer.writerow(['segment_id', 'path', 'transcript', 'speaker', 'duration', 'original_full_path'])
        
        # Iterate over the dataset and write each row to the CSV file
        for sample in tqdm(dataset):
            segment_id = sample['segment_id']
            path = sample['audio']['path']
            transcript = sample['text']
            speaker = sample['speaker']
            duration = float(sample['end_time']) - float(sample['begin_time'])
            original_full_path = sample['original_full_path']
            
            csv_writer.writerow([segment_id, path, transcript, speaker, duration, original_full_path])
        

    print(f"CSV file has been created at {output_csv}")


def filter_dataset(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())
    print(df['original_full_path'].nunique())
    print(df['segment_id'].nunique())


if __name__ == "__main__":
    dataset_size = 's'
    download_to = '/home/thivt1/data/datasets/gigaspeech'
    # download(dataset_size, download_to)
    # dataset2csv(dataset_size=dataset_size, download_to=download_to)
    csv_path = '/home/thivt1/data/datasets/my_giga_s/full_gigaspeech_s.csv'
    filter_dataset(csv_path)
