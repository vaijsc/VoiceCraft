import pandas as pd
import os
import hashlib

def hash_filename(filepath):
    return hashlib.md5(filepath.encode()).hexdigest()

def get_segment_id_from_path(path): 
    '''
    input: 
        path: 'big_processed_data/Khương_Ngọc_Đình/Hồi_Ký_Của_Các_Tướng_Tá_Sài_Gòn/tuong-ta-3/chunk-4423_22-4426_94.wav'
    output: 
        segment_id from path
    '''
    return "-".join(path.split('/')[3:]).replace('.wav', '').strip()

# load validation file
val_path = '/lustre/scratch/client/vinai/users/thivt1/code/zstts/split_long_audio/metadata/step20_val.csv'
val_data = pd.read_csv(val_path, sep='|', header=None, names=['audio_path', 'transcript', 'speaker', 'duration'])
val_data = val_data.values.tolist()
val_paths = [item[0] for item in val_data]

# load train file
train_path = '/lustre/scratch/client/vinai/users/thivt1/code/zstts/split_long_audio/metadata/step20_train.csv'
train_data = pd.read_csv(train_path, sep='|', header=None, names=['audio_path', 'transcript', 'speaker', 'duration'])
train_data = train_data.values.tolist()
train_paths = [item[0] for item in train_data]

train_segment_old2new = {get_segment_id_from_path(path): hash_filename(path) for path in train_paths}
val_segment_old2new = {get_segment_id_from_path(path): hash_filename(path) for path in val_paths}

enc_folder = '/home/thivt1/data/datasets/1,5khr_sachnoi/encodec_16khz_4codebooks'
phn_folder = '/home/thivt1/data/datasets/1,5khr_sachnoi/phonemes'

enc_files = os.listdir(enc_folder)
phn_files = os.listdir(phn_folder)

# files that are in train but not in enc
train_not_in_enc = set(train_segment_old2new.keys()) - set([os.path.splitext(file)[0] for file in enc_files])
val_not_in_enc = set(val_segment_old2new.keys()) - set([os.path.splitext(file)[0] for file in enc_files])
not_in_enc = train_not_in_enc.union(val_not_in_enc)

train_not_in_phn = set(train_segment_old2new.keys()) - set([os.path.splitext(file)[0] for file in phn_files])
val_not_in_phn = set(val_segment_old2new.keys()) - set([os.path.splitext(file)[0] for file in phn_files])
not_in_phn = train_not_in_phn.union(val_not_in_phn)

# # save files that are not in enc
data = train_data + val_data
# not_in_enc_data = [item for item in data if get_segment_id_from_path(item[0]) in not_in_enc]
# not_in_phn_data = [item for item in data if get_segment_id_from_path(item[0]) in not_in_phn]
# print(len(not_in_enc_data))
# print(len(not_in_phn_data))
# with open("not_in_enc.txt", "w") as f:
#     for item in not_in_enc_data:
#         f.write(f"{item[0]}|{item[1]}|{item[2]}|{item[3]}\n")

# list files in data that are in enc 
new_phn_folder = phn_folder.replace("1,5khr_sachnoi", 'sachnoi_1500hr')
new_enc_folder = enc_folder.replace("1,5khr_sachnoi", 'sachnoi_1500hr')
enc_files = set([os.path.splitext(file)[0] for file in enc_files])
enc_data_segment_id = [get_segment_id_from_path(item[0]) for item in data if get_segment_id_from_path(item[0]) in enc_files]
for old_segment_id in enc_data_segment_id:
    new_segment_id = train_segment_old2new.get(old_segment_id, val_segment_old2new.get(old_segment_id))
    old_phn_path = os.path.join(phn_folder, f"{old_segment_id}.txt")
    new_phn_path = os.path.join(new_phn_folder, f"{new_segment_id}.txt")

    old_enc_path = os.path.join(enc_folder, f"{old_segment_id}.txt")
    new_enc_path = os.path.join(new_enc_folder, f"{new_segment_id}.txt")
    breakpoint()

    # copy phn file
    os.system(f"cp {old_phn_path} {new_phn_path}")
    # copy enc file
    os.system(f"cp {old_enc_path} {new_enc_path}")

