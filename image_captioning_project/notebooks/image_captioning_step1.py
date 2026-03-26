import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Path to images
IMAGE_DIR = "../dataset/Images"

# Load VGG16 model (remove last classification layer)
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.inputs,
              outputs=base_model.layers[-2].output)

print("VGG16 loaded successfully")

# Extract features
features = {}

for img_name in tqdm(os.listdir(IMAGE_DIR)):
    img_path = os.path.join(IMAGE_DIR, img_name)

    # Load image
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Get feature vector
    feature = model.predict(image, verbose=0)

    # Store using image id (without extension)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

# Save features
np.save("../models/image_features.npy", features)
print("Image features extracted and saved")
