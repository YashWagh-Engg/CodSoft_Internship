import numpy as np
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load captions
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            image_id = tokens[0].split('.')[0]
            caption = ' '.join(tokens[1:])
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions


def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)
    for key, caps in captions.items():
        for i in range(len(caps)):
            cap = caps[i].lower()
            cap = cap.translate(table)
            cap = cap.replace('  ', ' ')
            caps[i] = 'startseq ' + cap + ' endseq'


# Load & clean captions
captions = load_captions(r"D:\CodSoft Internship\image_captioning_project\Dataset\captions.txt")
clean_captions(captions)

# Flatten captions
all_captions = []
for caps in captions.values():
    all_captions.extend(caps)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)

print("Vocabulary Size:", vocab_size)
print("Max Caption Length:", max_length)

# Save tokenizer and max_length
np.save("../models/tokenizer.npy", tokenizer)
np.save("../models/max_length.npy", max_length)
