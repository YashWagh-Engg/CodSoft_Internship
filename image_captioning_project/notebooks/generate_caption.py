import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences# type: ignore

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore

from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

from tensorflow.keras.models import Model # type: ignore

from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add # type: ignore

import matplotlib.pyplot as plt

print("STEP 1: Script started")

# ---------- Load tokenizer & config ----------
tokenizer = np.load("D:\Task 1\image_captioning_project\models/tokenizer.npy", allow_pickle=True).item()
max_length = int(np.load("D:\Task 1\image_captioning_project\models/max_length.npy"))
vocab_size = len(tokenizer.word_index) + 1
print("STEP 2: Tokenizer loaded")

# ---------- Rebuild model ----------
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
model.load_weights("../models/caption_weights.weights.h5")
print("STEP 3: Model built and weights loaded")

# ---------- VGG16 feature extractor ----------
vgg = VGG16(weights="imagenet")
vgg_model = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
print("STEP 4: VGG16 loaded")

# ---------- Helper functions ----------
def extract_feature(image_path):
    print("STEP 5: Extracting image feature")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, feature, max_length, temperature=0.8):
    in_text = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        preds = model.predict([feature, seq], verbose=0)[0]

        # Temperature scaling
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        next_word_id = np.random.choice(len(preds), p=preds)
        word = idx_to_word(next_word_id, tokenizer)

        if word is None:
            break

        in_text += " " + word
        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()


# ---------- Test on one image ----------
image_path = r"D:\Task 1\image_captioning_project\Dataset/images/1319634306_816f21677f.jpg"
feature = extract_feature(image_path)
caption = generate_caption(model, tokenizer, feature, max_length)

print("STEP 7: Caption generated")
print("Generated Caption:", caption)

# Show image
img = load_img(image_path)
plt.imshow(img)
plt.axis("off")
plt.show()
