import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load preprocessed data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
labels = np.load("labels.npy")

# Basic info
print("🔍 EDA Summary")
print(f"Total Images: {len(X_train) + len(X_test)}")
print(f"Train Images: {len(X_train)}")
print(f"Test Images: {len(X_test)}")
print(f"Image Shape: {X_train[0].shape}")
print(f"Number of Classes: {len(labels)}")
print(f"Labels: {labels}")

# 🔹 1. Count of Images per Class
train_counts = [np.sum(y_train == i) for i in range(len(labels))]
test_counts = [np.sum(y_test == i) for i in range(len(labels))]

plt.figure(figsize=(10, 5))
bar1 = plt.bar(labels, train_counts, label='Train', alpha=0.7)
bar2 = plt.bar(labels, test_counts, label='Test',
               alpha=0.7, bottom=train_counts)
plt.title("Images per Person (Train + Test)")
plt.xlabel("Person ID")
plt.ylabel("Number of Images")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("eda_bar_chart.png")  # saves the figure
plt.show()

# 🔹 2. Display Sample Faces per Class
print("🖼️ Showing 1 sample face per person:")
SAMPLES_PER_ROW = 6
rows = int(np.ceil(len(labels) / SAMPLES_PER_ROW))

fig, axes = plt.subplots(rows, SAMPLES_PER_ROW, figsize=(15, 2.5 * rows))
axes = axes.flatten()

for i in range(len(labels)):
    idx = np.where(y_train == i)[0][0]  # get first image index for this label
    img = (X_train[idx] * 255).astype(np.uint8)
    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"ID: {labels[i]}")
    axes[i].axis('off')

for j in range(len(labels), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("eda_face_grid.png")
plt.show()
