'''
cut audio prompt to 3s segment for inference of valle 
'''


from pydub import AudioSegment
from tqdm import tqdm
import os
import pandas as pd
import glob
import json
import librosa 


def find_cut_off_sec(filepath, transcript):
    print(f'file being processed: {filepath}')

    # copy audio file to prompt folder
    # shutil.copy(filepath, prompt_folder)

    # # save transcript to file
    filename = os.path.splitext(os.path.basename(filepath))[0]
    # with open(f"{prompt_folder}/{filename}.txt", 'w') as f:
    #     f.write(transcript)

    # get mfa alignment
    mfa_align_path = f"{prompt_folder}/{filename}.csv"
    print(f'mfa align path: {mfa_align_path}')
    beam_size = 500
    retry_beam_size = 2000
    if not os.path.isfile(mfa_align_path):
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
            cutoff_word_idx = i

    if silence_pos: 
        print(f'max silence duration: {max_silence_dur}')
        print(f'silence position: {silence_pos}')
        print(
            f"that corresponds to word {alignment.at[cutoff_word_idx, 'Label']} and {alignment.at[cutoff_word_idx + 1, 'Label']}")
        print(f'at line {cutoff_word_idx}')
        # cut_off_sec = (silence_pos[0] + silence_pos[1]) / 2
        cut_off_sec = silence_pos[0] + (silence_pos[1] - silence_pos[0]) / 3
        print(f'cut off sec: {cut_off_sec}')
    else: 
        wav_path = mfa_align_path.replace('.csv', '.wav')
        dur = librosa.get_duration(filename=wav_path)
        print(f'dur: {dur}')
        cut_off_sec = (ends[-1] + dur) / 2
        cutoff_word_idx = len(begins) - 1
        print(f'cutoff sec: {cut_off_sec}')
        print(f'cutoff word: {cutoff_word_idx}')
    return cut_off_sec, cutoff_word_idx


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


if __name__ == '__main__':
    # load test set
    json_files = sorted(glob.glob("testset_final/*.json"))
    # json_files = [file for file in json_files if "sachnoi-seen" in file or "vivoice" in file]
    print(f'there are {len(json_files)} json files in the test set')
    print(f'first 3 files: {json_files[:3]}')

    for i, speaker in enumerate(tqdm(json_files)):
        print(f'processing speaker {i}: {speaker}')
        with open(speaker, 'r') as f:
            data = json.load(f)

        speaker_name = speaker.split("/")[-1].replace(".json", "")
        print(f'speaker: {speaker_name}')

        # create folder for prompt data
        prompt_folder = f"testset_output_0910/{speaker_name}/audio_prompt"
        # os.makedirs(prompt_folder, exist_ok=True)

        prompt_data = data[0]
        original_transcript = prompt_data['transcript']
        print(f'prompt audio: {prompt_data}')

        audio_fn = prompt_data['path']
        # find cut off sec
        cut_off_sec, cut_off_word_idx = find_cut_off_sec(
            audio_fn, original_transcript)
        # margin = 0.04
        # cutoff_tolerance = 1 
        # cut_off_sec, cut_off_word_idx = find_closest_word_boundary(
        #     mfa_align_path, cut_off_sec, margin, cutoff_tolerance)
        print(f'cut off sec: {cut_off_sec}')
        print(f'cut off word index: {cut_off_word_idx}')

        # cut audio prompt based of cut off sec
        audio = AudioSegment.from_file(audio_fn)

        # Trim the first 3 seconds
        trimmed_audio = audio[:int(cut_off_sec * 1000)]  # Time in milliseconds

        # Export the trimmed audio
        cut_path = f"audio_prompt/{speaker_name}/audio_prompt.wav"
        os.makedirs(os.path.dirname(cut_path), exist_ok=True)
        trimmed_audio.export(cut_path, format="wav")
        
        # save transcript of corresponding audio cut
        ref_text = " ".join(original_transcript.split(" ")
                            [:cut_off_word_idx+1])
        transcript_path = f"audio_prompt/{speaker_name}/transcript.txt"
        with open(transcript_path, 'w') as f:
            f.write(ref_text)

        # save the cut off sec
        cut_off_sec_path = f"audio_prompt/{speaker_name}/cut_off_sec.txt"
        with open(cut_off_sec_path, 'w') as f:
            f.write(str(cut_off_sec))
        