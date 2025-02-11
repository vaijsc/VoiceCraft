import os
import torch
import string
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import sys
import glob
import json


#----MBART-----
model_name_or_path = "facebook/mbart-large-50"
output_dir = "/lustre/scratch/client/vinai/users/thivt1/code/oneshot/norm_text/output_mbart_text_norm4_fp16"
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(output_dir).to(device)


def infer_sample(input_sentence): 
    # Tokenize the input sentence
    inputs = tokenizer(input_sentence, return_tensors="pt",
                    padding=True, truncation=True, max_length=1024).to(device)

    # Generate the translation
    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=1024, num_beams=5)

    # Decode the generated tokens to text
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    print(f'input: {input_sentence}\n')
    print(f'normalized transcript: {translation}\n')


def create_batches(data, bs):
    batches = []
    for i in range(0, len(data), bs):
        batches.append(data[i:i + bs])
    return batches


def infer_big_dataset(): 
    json_files = sorted(glob.glob("json_files_vivoice_test/*.json"))
    print(f'there are {len(json_files)} json files')
    print(f'first 3 json files: {json_files[:3]}')
    for file in json_files: 
        process_jsonfile(file)
    

def process_jsonfile(jsonfile):
    # load the json file
    with open(jsonfile, 'r') as f:
        data = json.load(f)
        
    print(f'new len(data):', len(data))

    # create batches out of data
    input_sentences = [item['transcript'] for item in data]
    batch_size = 80
    batches = create_batches(input_sentences, batch_size)

    output_sentences = []
    for batch in tqdm(batches, total=len(batches)):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

        # Generate translations for the batch
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, max_length=1024, num_beams=5)

        # Decode the generated tokens to text
        norm_trans = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        output_sentences.extend(norm_trans)

    print('=====================================================')
    print('TEXT WITH ENGLISH')
    for tran, norm_tran in list(zip(input_sentences, output_sentences))[:10]:
        print(f'transcript: {tran}')
        print(f'normalized transcript: {norm_tran}\n')

    # write the normalized transcript to new json files
    output_jsonfile = jsonfile.replace("json_files_vivoice_test", "json_files_vivoice_test_norm")
    os.makedirs(os.path.dirname(output_jsonfile), exist_ok=True)
    with open(output_jsonfile, 'w') as f:
        for i, item in enumerate(data): 
            item['transcript'] = output_sentences[i]
            item['path'] = item['path'].replace("Voicecraft", "VoiceCraft")
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__': 
    # gpu_id = sys.argv[1]
    infer_big_dataset()
