# predict_image.py

import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import argparse

# --- 1. Argument Parsing ---
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# --- 2. Define Constants & Load Model ---
IMG_SIZE = 60 # Must be the same size as used in training
MODEL_PATH = 'cat_dog_model.h5'

print("✅ Loading the trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- 3. Load and Preprocess the Image ---
# Load the image in grayscale
img_array = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
# Resize it to the required 60x60 size
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# Reshape for the model and normalize
processed_image = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

# --- 4. Make a Prediction ---
prediction = model.predict([processed_image])[0]
labels = ['Cat', 'Dog']
# The output is an array of probabilities, e.g., [0.98, 0.02]
# We find the index with the highest probability
predicted_label = labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"🧠 Prediction: This is a {predicted_label} ({confidence:.2f}%)")

# --- 5. Display the Result on the Image ---
# Read the original color image for display
display_image = cv2.imread(args["image"])
display_text = f"{predicted_label}: {confidence:.2f}%"
# Put text on the image
cv2.putText(display_image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Prediction", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
