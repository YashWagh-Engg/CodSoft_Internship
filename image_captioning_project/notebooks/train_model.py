import tensorflow as tf
import numpy as np
import string
from tensorflow.keras.models import Model # type: ignore

from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add # type: ignore

from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

from tensorflow.keras.utils import to_categorical # type: ignore

from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

# ---- SELECT 5 IMAGE IDS (NO .jpg) ----
selected_images = [
    "1319634306_816f21677f", "1321723162_9d4c78b8af", "1321949151_77b77b4617", "1333888922_26f15c18c3", "1334892555_1beff092c3"   # <-- replace with YOUR real image IDs
]
# ---------- Load saved assets ----------
# Load features
features = np.load(
    r"D:\CodSoft Internship\image_captioning_project\models\image_features.npy",
    allow_pickle=True
).item()

# Filter features
features = {k: v for k, v in features.items() if k in selected_images}


tokenizer = np.load(
    r"D:\CodSoft Internship\image_captioning_project\models\tokenizer.npy",
    allow_pickle=True
).item()
max_length = int(np.load(
    r"D:\CodSoft Internship\image_captioning_project\models\max_length.npy"
))
vocab_size = len(tokenizer.word_index) + 1

# ---------- Load & clean captions ----------
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            image_id = tokens[0].split('.')[0]
            caption = ' '.join(tokens[1:])
            captions.setdefault(image_id, []).append(caption)
    return captions

def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)
    for k, caps in captions.items():
        for i in range(len(caps)):
            c = caps[i].lower().translate(table).replace('  ', ' ')
            caps[i] = 'startseq ' + c + ' endseq'

captions = load_captions(
    r"D:\CodSoft Internship\image_captioning_project\Dataset\captions.txt"
)
clean_captions(captions)

# Filter captions
captions = {k: v for k, v in captions.items() if k in selected_images}




captions = {k: v for k, v in captions.items() if k in selected_images}

# ---------- Data generator ----------
def data_generator(captions, features, tokenizer, max_length, vocab_size):
    while True:
        for img_id, caps in captions.items():
            if img_id not in features:
                continue

            feature = features[img_id][0]  # shape: (4096,)

            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]

                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]  # (max_length,)
                    out_seq = to_categorical(out_seq, num_classes=vocab_size)  # (vocab_size,)

                    yield ((feature, in_seq), out_seq)


# ---------- Model ----------
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder = add([fe2, se3])
decoder = Dense(256, activation='relu')(decoder)
outputs = Dense(vocab_size, activation='softmax')(decoder)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

print(model.summary())

# ---------- Train ----------
import tensorflow as tf

steps = sum(len(c) for c in captions.values())

dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(captions, features, tokenizer, max_length, vocab_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(4096,), dtype=tf.float32),
            tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        ),
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
)

# THIS LINE IS THE KEY
dataset = dataset.batch(1)

model.fit(
    dataset,
    steps_per_epoch=100,
    epochs=20,
    verbose=1
)

import os
os.makedirs("../models", exist_ok=True)

model.save_weights("../models/caption_weights.weights.h5")
print("Weights saved successfully")
