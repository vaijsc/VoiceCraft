import os
import json
import pandas as pd
import random


def create_subset_dataset(df, min_duration=0.3, max_duration=1.0):
    # Convert duration from seconds to hours
    df['duration_hours'] = df['duration'] / 3600

    # Group by speaker and calculate total duration
    grouped = df.groupby('speaker')['duration_hours'].sum().reset_index()

    # Filter speakers based on the duration range
    valid_speakers = grouped[(grouped['duration_hours'] >= min_duration) & (
        grouped['duration_hours'] <= max_duration)]['speaker']
    filtered_df = df[df['speaker'].isin(valid_speakers)]

    # Handle speakers with more than max_duration hours of audio
    excessive_speakers = grouped[grouped['duration_hours']
                                 > max_duration]['speaker']

    for speaker in excessive_speakers:
        speaker_df = df[df['speaker'] == speaker]
        total_duration = speaker_df['duration_hours'].sum()

        while total_duration > max_duration:
            # Randomly remove a file
            to_remove = random.choice(speaker_df.index)
            total_duration -= speaker_df.loc[to_remove, 'duration_hours']
            speaker_df = speaker_df.drop(to_remove)

        filtered_df = pd.concat([filtered_df, speaker_df])

    return filtered_df


metadata_path = "step14_tone_norm_transcript_no_multispeaker.txt"
cols = ['filename', 'transcript', 'speaker', 'duration']
df = pd.read_csv(metadata_path, sep='|', names=cols, header=None)
test_speakers = [
    'Huỳnh_Minh_Hiền', 'Lê_Á_Thi', 'Hoàng_Tín', 'Chủ_Tịch_Hồ_Chí_Minh',
    'Nguyễn_Đình_Khánh', 'Thanh_Vân', '50_Nghệ_Sĩ-27-Huu Chau', 'BBC',
    'Thy_Lan', 'Nam_Anh', 'Nguyễn_Ngọc', '50_Nghệ_Sĩ-15-Ly Hung',
    'Thích_Chân_Tính', 'Hoàng_Mến', '50_Nghệ_Sĩ-39-Thai Hoa', '50_Nghệ_Sĩ',
    '50_Nghệ_Sĩ-10-Do Trung Quan', 'Hải_Khuê', '50_Nghệ_Sĩ-36-Tang Thanh Ha',
    'Lê_Bảo_Quốc'
]

test_df = df[df['speaker'].isin(test_speakers)]

train_df = df[~df['speaker'].isin(test_speakers)]

filtered_train = create_subset_dataset(
    train_df, min_duration=0.0, max_duration=4)


train_path = "sach_noi_train.json"
test_path = 'sach_noi_test.json'
root_dir = "/lustre/scratch/client/vinai/users/thivt1/code/oneshot"

# Load dialect information from JSONL file
dialect_file = "/workspace/oneshot/data_stories_large_model.jsonl"
dialect_info = {}

with open(dialect_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        dialect_info[data['path']] = data['dialect']['Label']

# Function to create dictionary from DataFrame row


def create_dict(row):
    path = os.path.join(root_dir, row['filename'])
    dialect = dialect_info.get(path)
    if dialect:
        return {
            "path": path,
            "transcript": row['transcript'],
            "speaker": row['speaker'],
            "duration": row['duration'],
            "dialect": dialect,
            "segment_id": get_segment_id_from_path(row['filename'])
        }
    else:
        raise ValueError(f"Dialect information not found for {path}")


def get_segment_id_from_path(path):
    if len(path.split("/")[1:]) == 3:
        return "___".join(path.replace('.wav', '').split("/")[1:])
    else:
        speaker, book, chapter, segment = path.replace(
            '.wav', '').split("/")[1:]
        return f"{speaker}___{chapter}___{segment}"


# Create list of dictionaries for train and test sets
train_list = [create_dict(row) for index, row in filtered_train.iterrows()]
test_list = [create_dict(row) for index, row in test_df.iterrows()]

# Save to JSON files
with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_list, f, ensure_ascii=False, indent=4)

with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_list, f, ensure_ascii=False, indent=4)

print("JSON files created successfully.")
