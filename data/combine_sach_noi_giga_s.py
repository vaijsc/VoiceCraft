'''
Combine custom gigaspeech s dataset with sach noi dataset
'''


from datasets import load_from_disk, DatasetDict, concatenate_datasets


def commbine_dataset():
    # Load the datasets from local paths
    sach_noi = load_from_disk('/home/thivt1/data/datasets/sach_noi/subset_446hrs')
    gigas = load_from_disk('/home/thivt1/data/datasets/my_giga_s/113_hours_dataset')

    # Combine the training sets
    combined_train = concatenate_datasets([sach_noi['train'], gigas['train']])

    # Combine the evaluation sets
    combined_eval = concatenate_datasets([sach_noi['validation'], gigas['evaluation']])

    # Create a combined DatasetDict
    combined_dataset_dict = DatasetDict({
        'train': combined_train,
        'validation': combined_eval
    })

    # Save the combined dataset to a local folder
    combined_dataset_dict.save_to_disk(out_dir)
    print(f"Combined dataset saved to {out_dir}")

    
def test_combined_dataset():
    dataset = load_from_disk(out_dir)
    print(dataset)
    print(dataset['train'][0])
    
    
if __name__ == '__main__':
    out_dir = '/home/thivt1/data/datasets/sach_noi/combined_dataset'
    # commbine_dataset()
    # test_combined_dataset()

    sach_noi_path = "/home/thivt1/data/datasets/sach_noi/subset_446hrs"
    sach_noi = load_from_disk(sach_noi_path)
    print(sach_noi['train'][10])

    gigaspeech_path = "/home/thivt1/data/datasets/my_giga_s/113_hours_dataset"
    giga  = load_from_disk(gigaspeech_path)
    print(giga['train'][10])
