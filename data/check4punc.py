'''
voicecraft only handles 4 punctuations: . , ? !
This script checks if the input text of sach_noi contains any other punctuations
'''

from string import punctuation


if __name__ == "__main__":
    punc = set()
    with open('sach_noi_subset_110hrs_train_vc_yourtts.txt', 'r') as f:
        for line in f:
            path, text, speaker, duration = line.strip().split('|')
            for char in text:
                if char in punctuation and char not in ['.', ',', '?', '!']:
                    punc.add(char)

    print(punc)