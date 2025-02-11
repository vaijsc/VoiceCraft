import os
import shutil
import json
import glob 
from tqdm import tqdm


# List of source files
json_files = glob.glob("./testset_final/*.json")
json_files = [file for file in json_files if "sachnoi" in file]
print(f'there are {len(json_files)} json files in the test set')
print(json_files)
wav_files = []
for speaker in json_files: 
    with open(speaker, 'r') as f:
        data = json.load(f)

    for sample in data: 
        path = sample['path']
        wav_files.append(path)

print(f'there are {len(wav_files)} wav files in the test set')

# Base source and destination directories
# src_base = "/lustre/scratch/client/vinai/users/linhnt140/zero-shot-tts/preprocess_audio/vin27_16k"
# dst_base = "/lustre/scratch/client/vinai/users/thivt1/code/VoiceCraft/vin27_testset"
src_base = "/lustre/scratch/client/vinai/users/thivt1/code/oneshot/"
dst_base = "/lustre/scratch/client/vinai/users/thivt1/data/sachnoi_testset"

for file in tqdm(wav_files):
    # Determine the relative path from the source base
    relative_path = os.path.relpath(file, src_base)
    
    # Create the destination directory path
    dst_file = os.path.join(dst_base, relative_path)
    
    # Create any necessary directories in the destination path
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    
    # Copy the file to the new destination
    shutil.copy2(file, dst_file)

print("Files copied successfully.")
