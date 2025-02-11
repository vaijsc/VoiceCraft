'''
do batch inference on a list of speaker, 
make each speaker speaker several sentences
'''


import librosa
from tqdm import tqdm
import os
import pandas as pd
import glob
import json
import shutil
from inference_tts_scale import inference_one_sample
from models import voicecraft
from data.tokenizer import (
    tokenize_text,
    AudioTokenizer,
    ViTextTokenizer,
)
import random
import numpy as np
import torchaudio
import torch
import argparse


def find_cut_off_sec(filepath, transcript):
    print(f'file being processed: {filepath}')

    # copy audio file to prompt folder
    shutil.copy(filepath, prompt_folder)

    # save transcript to file
    filename = os.path.splitext(os.path.basename(filepath))[0]
    with open(f"{prompt_folder}/{filename}.txt", 'w') as f:
        f.write(transcript)

    # get mfa alignment
    mfa_align_path = f"{prompt_folder}/{filename}.csv"
    print(f'mfa align path: {mfa_align_path}')
    beam_size = 500
    retry_beam_size = 2000
    if not os.path.isfile(mfa_align_path):
    #     os.system(f"mfa align -v --clean -j 1 --output_format csv '{prompt_folder}' \
    #             viIPA hp_dtn_acoustic '{prompt_folder}' --beam {beam_size} --retry_beam {retry_beam_size}")
        os.system(f"mfa align -v --clean -j 1 --output_format csv '{prompt_folder}' \
                vietnamese_hanoi_mfa vietnamese_mfa '{prompt_folder}' --beam {beam_size} --retry_beam {retry_beam_size}")

    # load the alignment csv file
    alignment = pd.read_csv(mfa_align_path)
    alignment = alignment[alignment['Type'] == 'words']
    begins = alignment['Begin'].tolist()
    ends = alignment['End'].tolist()
    assert len(begins) == len(ends)

    max_silence_dur = 0.1 
    silence_pos = None
    for i in range(len(begins) - 1):
        if ends[i] < 3:  # only consider cutoff position after 3 seconds
            continue
        if ends[i] > 8:
            break
        if begins[i+1] - ends[i] > max_silence_dur:
            max_silence_dur = begins[i+1] - ends[i]
            silence_pos = (ends[i], begins[i+1])
            cut_off_word_idx = i

    if silence_pos: 
        print(f'max silence duration: {max_silence_dur}')
        print(f'silence position: {silence_pos}')
        print(
            f"that corresponds to word {alignment.at[cut_off_word_idx, 'Label']} and {alignment.at[cut_off_word_idx + 1, 'Label']}")
        print(f'at line {cut_off_word_idx}')
        # cut_off_sec = (silence_pos[0] + silence_pos[1]) / 2
        cut_off_sec = silence_pos[0] + (silence_pos[1] - silence_pos[0]) / 3
        print(f'cut off sec: {cut_off_sec}')
    else: 
        wav_path = mfa_align_path.replace('.csv', '.wav')
        dur = librosa.get_duration(filename=wav_path)
        print(f'dur: {dur}')
        cut_off_sec = (ends[-1] + dur) / 2
        cut_off_word_idx = len(begins) - 1
        print(f'cutoff sec: {cut_off_sec}')
        print(f'cutoff word: {cut_off_word_idx}')

    return cut_off_sec, cut_off_word_idx


def load_custom_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    config = checkpoint['config']
    phn2num = checkpoint['phn2num']
    model = voicecraft.VoiceCraft(args=config)

    # Load the weights from the checkpoint
    model.load_state_dict(checkpoint['model'])
    return model, vars(config), phn2num


def find_closest_word_boundary(alignments, cut_off_sec, margin, cutoff_tolerance=1):
    with open(alignments, 'r') as file:
        # skip header
        next(file)
        cutoff_time = None
        cutoff_index = None
        cutoff_time_best = None
        cutoff_index_best = None
        lines = [l for l in file.readlines()]
        for i, line in enumerate(lines[:-1]):
            end = float(line.strip().split(',')[1])
            if end >= cut_off_sec and cutoff_time == None:
                cutoff_time = end
                cutoff_index = i
            if end >= cut_off_sec and end < cut_off_sec + cutoff_tolerance and float(lines[i+1].strip().split(',')[0]) - end >= margin:
                cutoff_time_best = end + margin * 2 / 3
                cutoff_index_best = i
                break
        if cutoff_time_best != None:
            cutoff_time = cutoff_time_best
            cutoff_index = cutoff_index_best
        return cutoff_time, cutoff_index


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# hyperparams
device = "cuda" if torch.cuda.is_available() else "cpu"

# dialect2lang = {
#     'north': 'vi',
#     'south': 'vi-vn-x-south',
#     'center': 'vi-vn-x-central'
# }

language = "vi"
silence_tokens = [1388, 1898, 131]
codec_audio_sr = 16000
codec_sr = 50
top_k = 0
top_p = 0.8
temperature = 1
kvcache = 1
stop_repetition = 1
sample_batch_size = 10 


if __name__ == '__main__':
    # set seed
    seed = 1 
    seed_everything(seed)

    # load model
    model_path = "experiments/big/e830M_ft_mixed_4cb_5k_hours_max_tok_25k_grad_50_lr_5e-5/best_bundle.pth"
    # model_path = "experiments/sach_noi/e830M_ft_vi_4cb_vc_400hrs_max_tok_20k_grad_50_lr_5e-5/best_bundle.pth"
    print(f'loading custom model at {model_path}')
    model, config, phn2num = load_custom_model(model_path)
    model.to(device)
    model.eval()
    print('done')

    # load encodec ckpt
    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(encodec_fn):
        os.system(
            f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O ./pretrained_models/encodec_4cb2048_giga.th")
    # will also put the neural codec model on gpu
    audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device)

    # load test set
    json_files = sorted(glob.glob("testset_final/*.json"))
    print(len(json_files))
    print(f'first 3 json files: {json_files[:3]}')
    output_folder = "testset_output_0910" 
    for i, speaker in enumerate(tqdm(json_files)):
        print(f'processing speaker {i}: {speaker}')
        with open(speaker, 'r') as f:
            data = json.load(f)

        speaker_name = speaker.split("/")[-1].replace(".json", "")
        print(f'speaker: {speaker_name}')

        # create folder for prompt data
        prompt_folder = f"{output_folder}/{speaker_name}/audio_prompt"
        os.makedirs(prompt_folder, exist_ok=True)

        prompt_data = data[0]
        original_transcript = prompt_data['transcript']
        print(f'prompt audio: {prompt_data}')

        audio_fn = prompt_data['path']
        # find cut off sec
        cut_off_sec, cut_off_word_idx = find_cut_off_sec(
            audio_fn, original_transcript)

        # cut_off_sec, cut_off_word_idx = find_closest_word_boundary(
        #     mfa_align_path, cut_off_sec, margin, cutoff_tolerance)
        print(f'cut off sec: {cut_off_sec}')
        print(f'cut off word index: {cut_off_word_idx}')

        # get prompt end frame
        audio_dur = prompt_data['duration']
        prompt_end_frame = int(cut_off_sec * codec_audio_sr)

        # load text tokenizer
        # dialect = prompt_data['dialect']
        vi_text_tokenizer = ViTextTokenizer(language="vi") # no matter the dialect, use phonemizer for north

        # get ref phn seq
        ref_text = " ".join(original_transcript.split(" ")
                            [:cut_off_word_idx+1])
        ref_phn = tokenize_text(vi_text_tokenizer, ref_text)
        ref_phn = [i.replace(' ', '_') for i in ref_phn]

        # ============= INFERENCE =============
        for target_data in data[1:]:
            # define dest file & folder
            # target_filepath = target_data['path'].replace("linhnt140/zero-shot-tts/preprocess_audio/vin27_16k", "thivt1/code/VoiceCraft/vin27_testset")
            target_filepath = target_data['path']
            target_fn = os.path.splitext(os.path.basename(target_filepath))[0]
            print(f'target_fn: {target_fn}')

            output_dir = f"{output_folder}/{speaker_name}/{target_fn}"

            seg_save_fn_gen = f"{output_dir}/voicecraft_5k.wav"
            # south and center dialects need to be infered again 
            # if dialect == 'north' and os.path.exists(seg_save_fn_gen):
            #     continue

            # os.makedirs(output_dir, exist_ok=True)
            # shutil.copy(target_filepath, f'{output_dir}/original.wav')

            print(f'target_data: {target_data}')
            target_transcript = target_data['transcript']
            # transcript_fn = f"{output_dir}/transcript.txt"
            # with open(transcript_fn, 'w') as f:
            #     f.write(target_transcript)

            # get target phn seq
            target_phn = tokenize_text(vi_text_tokenizer, target_transcript)
            target_phn = [i.replace(' ', '_') for i in target_phn]

            # conbine ref and target phn seq
            concat_target_transcript = ref_text + " " + target_transcript
            print(f'concat target_transcript: {concat_target_transcript}')
            phn_seq = ref_phn + ["_"] + target_phn
            print(f'phn_seq: {phn_seq}')

            # get result
            decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache,
                             "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}
            with torch.no_grad():
                concated_audio, gen_audio = inference_one_sample(model, argparse.Namespace(
                    **config), phn2num, phn_seq, audio_tokenizer, audio_fn, concat_target_transcript, device, decode_config, prompt_end_frame)

            concated_audio, gen_audio = concated_audio[0].cpu(
            ), gen_audio[0].cpu()

            # save result to file
            print(f'seg_save_fn_gen: {seg_save_fn_gen}')

            torchaudio.save(seg_save_fn_gen, gen_audio, codec_audio_sr)
            # NOTE: HOPE IT HELPS
            del concated_audio, gen_audio
            del phn_seq, target_phn
            torch.cuda.empty_cache() 
