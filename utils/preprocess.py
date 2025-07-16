import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter

DATASET_PATH = "dataset/"
IMG_SIZE = 128

X = []
y = []

# Only include numeric folder names as labels
label_names = sorted([
    name for name in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, name)) and name.isdigit()
], key=lambda x: int(x))

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        return img[y:y + h, x:x + w]
    return img  # fallback


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
        img = img / 255.0  # normalize
        X.append(img)
        y.append(label_index)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# EDA START 
print("\n--- EDA ---")
print(f"Total Samples: {len(X)}")
print(f"Image shape: {X[0].shape}")
print("Labels and Counts:")
label_counts = Counter(y)
for idx, count in label_counts.items():
    print(f"{label_names[idx]} ({idx}): {count} samples")

# Plot class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=y)
plt.xticks(ticks=np.arange(len(label_names)), labels=label_names, rotation=45)
plt.title("Class Distribution")
plt.xlabel("Labels")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# Show one sample face per class (converted to RGB)
fig, axs = plt.subplots(1, len(label_names), figsize=(15, 5))
for i, label in enumerate(label_names):
    idx = y.tolist().index(i)
    img_rgb = cv2.cvtColor((X[idx] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    axs[i].imshow(img_rgb)
    axs[i].set_title(label)
    axs[i].axis('off')
plt.suptitle("Sample Face from Each Class")
plt.tight_layout()
plt.show()

# Visualize average image (converted to RGB)
avg_img = np.mean(X, axis=0)
avg_img_rgb = cv2.cvtColor((avg_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
plt.imshow(avg_img_rgb)
plt.title("Average Face Image")
plt.axis('off')
plt.show()
#  EDA END

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save processed arrays
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("labels.npy", label_names)

print("\n✅ Done preprocessing with face cropping.")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
