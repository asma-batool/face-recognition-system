import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/" 
IMG_SIZE = 128

X = []
y = []
label_names = sorted(os.listdir(DATASET_PATH))

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        return img[y:y+h, x:x+w]  # return first detected face
    return img  # fallback: return original if no face found


for label_index, folder_name in enumerate(label_names):
    folder_path = os.path.join(DATASET_PATH, folder_name)
    if not os.path.isdir(folder_path):
        continue
    for filename in os.listdir(folder_path):
        if not filename.endswith(".jpg"):
            continue
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Crop face and resize
        img = crop_face(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # normalize
        X.append(img)
        y.append(label_index)

# Convert to NumPy
X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save arrays
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("labels.npy", label_names)

print(" Done preprocessing with face cropping.")
print(f"Total images: {len(X)}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Labels: {label_names}")