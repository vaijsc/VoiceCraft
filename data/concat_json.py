import json

# Define file paths
sach_noi_train_json = '/workspace/VoiceCraft/data/sach_noi_train.json'
sach_noi_test_json = '/workspace/VoiceCraft/data/sach_noi_test.json'

# This is actually a JSON file
vin27_json = '/workspace/TTS/recipes/vctk/yourtts/VIN27/full_metadata.csv'

# output path
output_json_path = '/workspace/VoiceCraft/data/combined_data.json'

# Function to load JSON files
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load data
train_data = load_json(sach_noi_train_json)
test_data = load_json(sach_noi_test_json)
metadata_data = load_json(vin27_json)

# Combine data
combined_data = train_data + test_data + metadata_data

# Save combined data to a new JSON file
with open(output_json_path, 'w', encoding='utf-8') as file:
    json.dump(combined_data, file, ensure_ascii=False, indent=4)

print(f'Combined data saved to {output_json_path}')
