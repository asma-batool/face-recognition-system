import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/"
IMG_SIZE = 128

X = []
y = []

label_names = sorted([
    name for name in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, name)) and name.isdigit()
], key=lambda x: int(x))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        return img[y:y + h, x:x + w]
    return img

# Load and preprocess images
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
        img = crop_face(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        X.append(img)
        y.append(label_index)

# Convert and split
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("labels.npy", label_names)

print("\n✅ Done preprocessing with face cropping.")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
