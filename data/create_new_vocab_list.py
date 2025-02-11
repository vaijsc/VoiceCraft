import torch
import os

if __name__ == '__main__':
    # Load the checkpoint
    ckpt_path = "../pretrained_models/830M_TTSEnhanced.pth"
    ckpt = torch.load(ckpt_path)
    
    # Check its phn2num dict 
    phn2num = ckpt['phn2num']
    old_vocab = list(phn2num.keys())  # Convert dict_keys to list
    breakpoint()
    
    # Load my new vocab file
    vocab_fn = "/home/thivt1/data/datasets/sachnoi_1500hr/dataset_vocab.txt"
    with open(vocab_fn, "r") as f:
        temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
        new_vocab = [item[1] for item in temp]

    print(f'len(old_vocab): {len(old_vocab)}')
    print(f'len(new_vocab): {len(new_vocab)}')
    print(f'len(old_vocab & new_vocab): {len(set(old_vocab) & set(new_vocab))}')
    print(f'len(old_vocab - new_vocab): {len(set(old_vocab) - set(new_vocab))}')
    print(f'len(new_vocab - old_vocab): {len(set(new_vocab) - set(old_vocab))}')
    print(f'new vocab introduced: {set(new_vocab) - set(old_vocab)}')
    
    # Combine the two vocab sets with old_vocab first
    combined_vocab = old_vocab.copy()
    combined_vocab += [phn for phn in new_vocab if phn not in old_vocab]

    # Save the new vocab file
    save_dir = os.path.join(os.path.dirname(vocab_fn), 'vocab.txt')
    with open(save_dir, "w") as f:
        for i, phn in enumerate(combined_vocab):
            f.write(f"{i} {phn}\n")

    print(f"Combined vocab saved to {save_dir}")