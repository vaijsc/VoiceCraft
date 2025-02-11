'''
create my own version of gigaspeech s dataset with 
half of audio duration (113 hours instead of 250 hours).

used to finetune voicecraft along with > 400 hours of 
vietnamese dataset (sach_noi)
'''


import pandas as pd
from datasets import Dataset, Audio, load_from_disk, DatasetDict


def create_dataset_from_metadata(metadata_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(metadata_file)

    # Prepare the data in the desired format
    data = {
        'segment_id': df['segment_id'].tolist(),
        'speaker': df['speaker'].astype(str).tolist(),
        'text': df['transcript'].tolist(),
        'audio': [{'path': path} for path in df['path'].tolist()],
        'duration': df['duration'].tolist()
    }

    print('0')

    # Convert the data to a Hugging Face dataset
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio())
    print('1')

    # Split the dataset into training and evaluation sets
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)

    print('2')

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })

    print('3')

    # Save the datasets to local folders
    save_dir = '/home/thivt1/data/datasets/my_giga_s/113_hours_dataset'
    dataset_dict.save_to_disk(save_dir)
    print(f"Datasets saved to {save_dir}")


def test_dataset():
    dataset = load_from_disk(
        '/home/thivt1/data/datasets/my_giga_s/113_hours_dataset')
    print(dataset)
    print(dataset['train'][0])


if __name__ == "__main__":
    create_dataset_from_metadata('subset_giga_s.csv')
    test_dataset()
