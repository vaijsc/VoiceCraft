import os
from tqdm import tqdm
from datasets import Dataset, Audio, load_from_disk, DatasetDict
from sklearn.model_selection import train_test_split


# Define a function to parse the metadata file and create the dataset structure
def create_dataset_from_metadata(metadata_file):
    data = {
        'segment_id': [],
        'speaker': [],
        'text': [],
        'audio': [],
        'duration': []
    }

    total_lines = sum(1 for _ in open(metadata_file))

    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines):
            # for line in tqdm(islice(f, 100), total=100):
            path, text, speaker, duration = line.strip().split('|')

            full_audio_path = os.path.join(
                "/workspace/oneshot", path.replace("wavs", "big_processed_data"))
            audio = {
                'path': full_audio_path,
                # 'array' and 'sampling_rate' are usually handled by the dataset loader,
                # so we just provide the path here.
            }
            segment_id = get_segment_id_from_path(path)
            data['segment_id'].append(segment_id)
            data['speaker'].append(speaker)
            data['text'].append(text)
            data['audio'].append(audio)
            data['duration'].append(float(duration))

    dataset = Dataset.from_dict(data)
    # Cast the audio column to the Audio feature type
    dataset = dataset.cast_column("audio", Audio())
    return dataset


def get_segment_id_from_path(path):
    if len(path.split("/")[1:]) == 3:
        return "___".join(path.replace('.wav', '').split("/")[1:])
    else: 
        speaker, book, chapter, segment = path.replace('.wav', '').split("/")[1:]
        return f"{speaker}___{chapter}___{segment}"


def split_dataset_by_speakers(dataset, test_size=0.1):
    # Get unique speakers
    speakers = list(set(dataset['speaker']))
    print(f"Total speakers: {len(speakers)}")

    # Split speakers into train and validation sets
    train_speakers, val_speakers = train_test_split(
        speakers, test_size=test_size, random_state=42)
    print(f'n_speakers in train set: {len(train_speakers)}')
    print(f'n_speakers in val set: {len(val_speakers)}')

    # Create train and validation datasets based on the speakers
    train_dataset = dataset.filter(
        lambda example: example['speaker'] in train_speakers)
    val_dataset = dataset.filter(
        lambda example: example['speaker'] in val_speakers)

    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def test_vi_dataset():
    dataset = load_from_disk(save_dir)
    print(dataset)
    print(dataset['train'][0])


if __name__ == "__main__":
    # ====== CREATE DATASET AND SAVE IN HUGGINGFACE FORMAT ======
    # Path to the metadata file
    metadata_file = 'filtered_sach_noi_0.1_1h.txt'
    # Create the dataset
    dataset = create_dataset_from_metadata(metadata_file)
    dataset_dict = split_dataset_by_speakers(dataset, test_size=0.05)

    # # ===== path before saving is absolute, after saving and loading from disk is relative =====
    # # print(dataset)
    # # print('before saving')
    # # print(dataset[0]['audio']['array'].dtype)
    # # print(dataset[0])

    # # print('after saving')
    # # dataset.save_to_disk('/home/thivt1/data/datasets/sach_noi/subset_40hrs')
    # # dataset = load_from_disk('/home/thivt1/data/datasets/sach_noi/subset_40hrs')
    # # print(dataset[0]['audio']['array'].dtype)
    # # print(dataset[0])
    # # ==================

    # Save the dataset dictionary to disk
    save_dir = '/home/thivt1/data/datasets/sach_noi/subset_446hrs'
    dataset_dict.save_to_disk(save_dir)

    # ====== TEST THE DATASET ======
    test_vi_dataset()
