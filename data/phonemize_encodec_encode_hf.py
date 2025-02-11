import argparse
import torchaudio 
import pandas as pd
import hashlib


def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument('--save_dir', type=str, default="/data/scratch/pyp/datasets/gigaspeech_phn_enc_manifest_debug", help="path to the manifest, phonemes, and encodec codes dirs")
    parser.add_argument('--encodec_model_path', type=str, default="/data/scratch/pyp/exp_pyp/audiocraft/encodec/xps/6f79c6a8/checkpoint.th")
    parser.add_argument('--n_workers', type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument('--mega_batch_size', type=int, default=100, help="Number of samples in each mega batch for multiprocess dataloading")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for encodec encoding, decrease it if OOM. This is the sum of batch size *over all gpus*, so increase it if you are using more gpus")
    parser.add_argument('--model_sr', type=int, default=16000, help='encodec input audio sample rate')
    parser.add_argument('--downsample_rate', type=int, default=320, help='encodec downsample rate')
    parser.add_argument('--model_code_sr', type=int, default=50, help='encodec model code sample rate')
    parser.add_argument('--len_cap', type=float, default=35.0, help='will drop audios that are longer than this number')
    parser.add_argument('--max_len', type=int, default=30000, help='max length of audio in samples, if exceed, will cut a batch into half to process, decrease this number if OOM on your machine')
    return parser.parse_args()


def get_segment_id_from_path(filepath): 
    '''
    input: 
        path: 'big_processed_data/Khương_Ngọc_Đình/Hồi_Ký_Của_Các_Tướng_Tá_Sài_Gòn/tuong-ta-3/chunk-4423_22-4426_94.wav'
    output: 
        segment_id from path
    '''
    return hashlib.md5(filepath.encode()).hexdigest()
    # return filepath.split('/')[3:].replace('.wav', '').strip()
    

if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()

    import os
    import numpy as np
    import torch
    import tqdm
    import time

    from tokenizer import ViTextTokenizer, tokenize_text
    
    # get the path
    phn_save_root = os.path.join(args.save_dir, "phonemes")
    codes_save_root = os.path.join(args.save_dir, "encodec_16khz_4codebooks")
    vocab_fn = os.path.join(args.save_dir, "dataset_vocab.txt")
    os.makedirs(phn_save_root, exist_ok=True)
    os.makedirs(codes_save_root, exist_ok=True)


    def sort_by_audio_len(lens):
        '''
        return indecies of lens sorted from biggest to smallest 
        '''
        inds = np.argsort(lens).tolist()
        logging.info(f"longest: {lens[inds[-1]]*args.model_code_sr} encodec codes, {lens[inds[-1]]:.2f} sec.")
        logging.info(f"shortest: {lens[inds[0]]*args.model_code_sr} encodec codes, {lens[inds[0]]:.2f} sec.")
        logging.info(f"median: {lens[inds[len(inds)//2]]*args.model_code_sr} encodec codes, {lens[inds[len(inds)//2]]:.2f} sec.")
        logging.info(f"95 percentile longest: {lens[inds[int(len(inds)*0.95)]]*args.model_code_sr} encodec codes, {lens[inds[int(len(inds)*0.95)]]:.2f} sec.")
        return inds[::-1] # revert the list
    
    def write_array_to_txt_file(array, filename):
        with open(filename, 'w') as f:
            for a in array[:-1]: # write from the first to the second last element
                f.write(' '.join(map(str, a))+'\n')
            f.write(' '.join(map(str, array[-1]))) # write the last element

    # phonemization
    # load tokenizer
    # load the encodec model
    from audiocraft.solvers import CompressionSolver
    print('loading encodec model...')
    model = CompressionSolver.model_from_checkpoint(args.encodec_model_path)
    model = model.cuda()
    model = model.eval()
    print('encodec model loaded')
    text_tokenizer = ViTextTokenizer(language='vi')

    # https://github.com/SpeechColab/GigaSpeech
    # there are only four different punctuations
    # need to check whether there are other < started strings

    punc2sym = {" <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?", " <EXCLAMATIONPOINT>": "!"} # note the space in front of each punc name
    gar2sym = {"<SIL>": "#%#", "<MUSIC>": "##%", "<NOISE>": "%%#", "<OTHER>":"%#%"} # so that they are savely keep as the original sym when using tokenize_text
    punc2sym.update(gar2sym)

    word2sym = { "h æ ʃ h ɐ ʃ p ɚ s ɛ n t": "<MUSIC>", "h æ ʃ p ɚ s ɛ n t h æ ʃ": "<SIL>", "p ɚ s ɛ n t h ɐ ʃ p ɚ s ɛ n t": "<OTHER>", "p ɚ s ɛ n t p ɚ s ɛ n t h æ ʃ": "<NOISE>"}
    forbidden_words = set(['#%#', '##%', '%%#', '%#%'])

    logging.info("loading the dataset...")
    stime = time.time()
    # train file 
    train_path = '/lustre/scratch/client/vinai/users/thivt1/code/zstts/split_long_audio/metadata/step20_train.csv'
    train_data = pd.read_csv(train_path, sep='|', header=None, names=['audio_path', 'transcript', 'speaker', 'duration'])
    train_data = train_data.values.tolist()
    print(f'len(train_data): {len(train_data)}')

    # val file 
    val_path = '/lustre/scratch/client/vinai/users/thivt1/code/zstts/split_long_audio/metadata/step20_val.csv'
    val_data = pd.read_csv(val_path, sep='|', header=None, names=['audio_path', 'transcript', 'speaker', 'duration'])
    val_data = val_data.values.tolist()
    print(f'len(val_data): {len(val_data)}')
    logging.info(f"time spend on loading the dataset: {time.time() - stime:.2f} seconds")
    splits = [train_data, val_data]

    # make sure after hash training and validation set, there is no overlap
    train_segment_ids = [get_segment_id_from_path(item[0]) for item in train_data]
    val_segment_ids = [get_segment_id_from_path(item[0]) for item in val_data]
    overlap = set(train_segment_ids) & set(val_segment_ids)
    assert len(overlap) == 0
    breakpoint()

    # logging.info(f"phonemizing...")
    phn_vocab = set()
    all_lens = []
    
    # PHONEMIZATION
    # you will see a ton of [WARNING] words_mismatch.py:88......, it's not an issue
    skip = 0
    for split in tqdm.tqdm(splits):
        for item in tqdm.tqdm(split):
            path, text, speaker, duration = item
            save_fn = os.path.join(phn_save_root, get_segment_id_from_path(path) + ".txt")
            if sum(word in forbidden_words for word in text.split(" ")):
                logging.info(f"skip {item['segment_id']}, because it contains forbiden words. It's transcript: {text}")
                skip += 1
                continue
            for k, v in punc2sym.items():
                text = text.replace(k, v)
        
            phn = tokenize_text(text_tokenizer, text)
            phn_seq = " ".join(phn)
            for k, v in word2sym.items():
                phn_seq = phn_seq.replace(k, v)
            phn_vocab.update(phn_seq.split(" "))
            all_lens.append(len(phn_seq.split(" ")))
            with open(save_fn, "w") as f:
                f.write(phn_seq)
        logging.info(f"split {split} has {len(split)} samples in total, skipped {skip} due to forbiden words")

    print(f"phn vocab size: {len(list(phn_vocab))}")
    print("phn sequence stats: ")
    print(f"longest: {max(all_lens)}")
    print(f"shortest: {min(all_lens)}")
    print(f"median: {np.quantile(all_lens, 0.5)}")
    print(f"95 percentile longest: {np.quantile(all_lens, 0.95)}")
    print("write vocabulary to ", vocab_fn)
    with open(vocab_fn, "w") as f:
        for i, phn in enumerate(list(phn_vocab)):
            if i < len(list(phn_vocab)) - 1:
                f.write(f"{str(i)} {phn}\n")
            else:
                f.write(f"{str(i)} {phn}")
    # ==============================================

    class mydataset(torch.utils.data.Dataset):
        def __init__(self, data):
            super().__init__()
            self.data = data
            self.root = '/lustre/scratch/client/vinai/users/thivt1/code/oneshot'
        def __len__(self):
            return len(self.data)
        def __getitem__(self, ind):
            try:
                item = self.data[ind]
                path, text, speaker, duration = item
                segment_id = get_segment_id_from_path(path)
                path = os.path.join(self.root, path)
                audio = torchaudio.load(path)[0].squeeze()
                sr = 16000
            except:
                print(f'fucking error while accessing data point elements')
                return None, None, None, None, None, None
            
            return segment_id, audio, sr, text, duration
        def collate(self, batch):
            res = {'segment_id': [], "audio": [], "sr": [], "transcript": [], "duration": []}
            for item in batch:
                if item[0] != None:
                    res['segment_id'].append(item[0])
                    res['audio'].append(item[1])
                    res['sr'].append(item[2])
                    res['transcript'].append(item[3])
                    res['duration'].append(item[4])
                else: 
                    print('item[0] is fucking none')
            return res


    ## encodec codes extraction
    logging.info("ENCODEC ENCODING...")
    train_dataset = mydataset(data=train_data) # dataset is not splitted into train & valid set yet 
    val_dataset = mydataset(data=val_data) # dataset is not splitted into train & valid set yet 
    splits = [train_dataset, val_dataset]
    loaders = [torch.torch.utils.data.DataLoader(dataset, 
                                                batch_size=args.mega_batch_size, 
                                                shuffle=False, 
                                                drop_last=False, 
                                                num_workers=args.n_workers, 
                                                collate_fn=dataset.collate)
              for dataset in splits]
    for split, loader in zip(splits, loaders):
        skip = 0
        # logging.info(f"now processing split {split}...")
        mega_n_steps = int(np.ceil(len(split) / args.mega_batch_size))
        logging.info(f"partition the dataset into {mega_n_steps} parts, each has {args.mega_batch_size} samples")
        for m, mega_batch in enumerate(loader):
            logging.info(f"====================================")
            logging.info(f"====================================")
            logging.info(f"now processing mega step {m+1}/{mega_n_steps}")
            print(f'len(mega_batch): {len(mega_batch)}')
            lengths = mega_batch['duration']

            sorted_inds = sort_by_audio_len(lengths) # [3,7,2,1,0]
            # skip lengths that are too long or too short
            for j in range(len(sorted_inds))[::-1]: # go from last index of sorted_inds -> traverse from shortest to longest length
                if lengths[sorted_inds[j]] < 0.2 or lengths[sorted_inds[j]] > args.len_cap: # skip samples that are too short (shorter than 0.2s), or too big (bigger than 80s)
                    skip += 1
                    del sorted_inds[j]
            
            n_steps = int(np.ceil(len(sorted_inds) / args.batch_size)) # n_steps = mega_batch_size / batch_size = 100/4 default
            for n in tqdm.tqdm(range(n_steps), disable=True): # process from the longest to shortest audio
                inds_used = sorted_inds[n*args.batch_size:(n+1)*args.batch_size] # the indices of this batch
                audio_batch = [mega_batch['audio'][id] for id in inds_used]
                sr_batch = [mega_batch['sr'][id] for id in inds_used]
                segment_id_batch = [mega_batch['segment_id'][id] for id in inds_used]
                text_batch = [mega_batch['transcript'][id] for id in inds_used]
                padded_wav = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True).unsqueeze(1) # [B, T] -> [B, 1, T]
                all_lens = [lengths[id] for id in inds_used]
                with torch.no_grad():
                    if max(all_lens) > args.max_len and len(all_lens) > 1: # NOTE decrease args.max_len if OOM, or chunk it into more than 2 forward passes
                        codes = []
                        inwav = padded_wav.cuda()
                        codes.append(model.encode(inwav[:len(inwav)//2])[0].cpu())
                        codes.append(model.encode(inwav[len(inwav)//2:])[0].cpu())
                        codes = torch.cat(codes, dim=0)
                    else:
                        encoded_frames = model.encode(padded_wav.cuda())
                        logging.info(f"encoded_frames: {encoded_frames[0].shape}")
                        codes = encoded_frames[0].cpu()
                
                for i, length in enumerate(all_lens):
                    save_fn = os.path.join(codes_save_root, segment_id_batch[i]+".txt")
                    # actual length != padded length, 
                    actual_len = round(length * args.model_code_sr) # n_seconds * freq (320)
                    if type(codes) == list: 
                        cur_code = codes[i].tolist()
                    else:
                        cur_code = codes[i, :, :actual_len].tolist()
                    write_array_to_txt_file(cur_code, save_fn)
