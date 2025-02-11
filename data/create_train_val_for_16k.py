
if __name__ == "__main__":
    # read train data for 16khz
    train_16k_path = "/home/thivt1/data/datasets/gigaspeech_xs_phn_enc_manifest/xs/manifest/train_with_test.txt"
    with open(train_16k_path, 'r') as f:
        train_16k_data = [line.strip().split("\t") for line in f.readlines()]

    # read validation data for 16khz
    val_16k_path = "/home/thivt1/data/datasets/gigaspeech_xs_phn_enc_manifest/xs/manifest/validation_with_test.txt"
    with open(val_16k_path, 'r') as f:
        val_16k_data = [line.strip().split("\t") for line in f.readlines()]

    # read train data for 24khz
    train_24k_path = "/home/thivt1/data/datasets/gigaspeech_phn_enc_manifest_8cb/xs/manifest/train.txt"
    with open(train_24k_path, 'r') as f:
        train_24k_filenames = [line.strip().split("\t")[1] for line in f.readlines()]

    # read validation data for 24khz
    val_24k_path = "/home/thivt1/data/datasets/gigaspeech_phn_enc_manifest_8cb/xs/manifest/validation.txt"
    with open(val_24k_path, 'r') as f:
        val_24k_filenames = [line.strip().split("\t")[1] for line in f.readlines()]
        
    # create train set for 16khz
    train_16k = []
    for line in train_16k_data:
        if line[1] in train_24k_filenames:
            train_16k.append(line)
            
    # create validation set for 16khz
    val_16k = []
    for line in val_16k_data:
        if line[1] in val_24k_filenames:
            val_16k.append(line)

    # write train set for 16khz
    train_16k_output_path = "/home/thivt1/data/datasets/gigaspeech_xs_phn_enc_manifest/xs/manifest/train.txt"
    with open(train_16k_output_path, 'w') as f:
        for line in train_16k:
            f.write("\t".join(line) + "\n")
    
    # write validation set for 16khz
    val_16k_output_path = "/home/thivt1/data/datasets/gigaspeech_xs_phn_enc_manifest/xs/manifest/validation.txt"
    with open(val_16k_output_path, 'w') as f:
        for line in val_16k:
            f.write("\t".join(line) + "\n")
            