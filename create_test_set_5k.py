'''
create test set for the 5k hour dataset:
- 20 speakers that speak the least in sachnoi
- 63 speakers from vin27 (one from each province)
'''


import json


test_set = {}

# load test set from sachnoi
with open("data/sach_noi_test.json", 'r') as f:
    sachnoi_test = json.load(f)

for sample in sachnoi_test:
    item = {
        'segment_id': sample['segment_id'],
        "path": sample["path"],
        'duration': sample['duration'],
        "transcript": sample["transcript"],
        'dialect': sample['dialect'],
    }
    speaker = f'sachnoi-{sample["speaker"]}'
    test_set.setdefault(speaker, []).append(item)

# load test set from vin27
with open("/lustre/scratch/client/vinai/users/thivt1/code/TTS/recipes/ljspeech/xtts_v2/VIN27/test.json", 'r') as f:
    vin27_test = json.load(f)

for sample in vin27_test:
    item = {
        'segment_id': sample['segment_id'],
        "path": sample["path"],
        'duration': sample['duration'],
        "transcript": sample["transcript"],
        'dialect': sample['dialect'],
    }
    speaker = f'vin27-{sample["speaker"]}'
    test_set.setdefault(speaker, []).append(item)

# save test set to file
print(f"Total speakers: {len(test_set)}")
for speaker, samples in test_set.items():
    print(f"{speaker}: {len(samples)} samples")
    # save each speaker as a json file
    with open(f"test_set_5k/{speaker}.json", 'w') as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)
