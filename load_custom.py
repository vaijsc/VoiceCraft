import torch
from models.voicecraft import VoiceCraft

def load_custom_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    # checkpoint.keys() = dict_keys(['model', 'optimizer', 
    #                       'scheduler', 'config', 'phn2num'])
    config = checkpoint['config']
    phn2num = checkpoint['phn2num']
    model = VoiceCraft(args=config)

    # Load the weights from the checkpoint
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, vars(config), phn2num


if __name__ == "__main__": 
    ckpt_path = 'experiments/gigaspeech/e830M_8cb_max_tok_50k_grad_50_lr_5e-3/best_bundle.pth'
    model, config, phn2num = load_custom_model(ckpt_path)
    print(model)
    print(config)
    print(phn2num)
    
