#!/bin/bash

# Directory containing the original audio files
input_dir="/home/thivt1/data/datasets/my_gigaspeech_dataset/audio_files_16k"
output_dir="/home/thivt1/data/datasets/my_gigaspeech_dataset/audio_files"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Target sampling rate: 24kHz
new_sample_rate=24000

# Loop over all audio files in the input directory
for input_file in "$input_dir"/*.wav; do
    # Get the base name of the file (without the directory path)
    base_name=$(basename "$input_file")
    
    # Define the output file path
    output_file="$output_dir/$base_name"
    
    # Resample the audio file using sox
    sox "$input_file" -r "$new_sample_rate" "$output_file"
done

echo "All files have been resampled to 24kHz."