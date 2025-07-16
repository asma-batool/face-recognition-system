import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_distances
import itertools

# Load preprocessed data
X = np.load("X_train.npy")
y = np.load("y_train.npy")
label_names = np.load("labels.npy", allow_pickle=True)

SAMPLE_SIZE = 150  

# 1️⃣ Correlation Matrix (Pixel-level Similarity)
def plot_correlation_matrix(X, sample_size=SAMPLE_SIZE):
    indices = random.sample(range(len(X)), sample_size)
    sample_images = X[indices].reshape(sample_size, -1)
    corr_matrix = np.corrcoef(sample_images)

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, cmap='coolwarm', cbar_kws={'label': 'Correlation'}, square=True)
    plt.title("Image Correlation Matrix (Pixel-Level)")
    plt.tight_layout()
    plt.show()

    # Top 5 correlated pairs
    pairs = list(itertools.combinations(range(sample_size), 2))
    correlations = [(i, j, corr_matrix[i, j]) for (i, j) in pairs]
    top_pairs = sorted(correlations, key=lambda x: -abs(x[2]))[:5]

    print("\n🔍 Top 5 Most Correlated Image Pairs:")
    for i, j, corr in top_pairs:
        print(f"Image {i} and {j} — Correlation: {corr:.4f}")

    # Show visual comparison
    fig, axs = plt.subplots(len(top_pairs), 2, figsize=(6, 10))
    for row, (i, j, corr) in enumerate(top_pairs):
        img1 = (X[indices[i]] * 255).astype(np.uint8)
        img2 = (X[indices[j]] * 255).astype(np.uint8)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        axs[row, 0].imshow(img1)
        axs[row, 0].set_title(f"Image {i}")
        axs[row, 0].axis('off')

        axs[row, 1].imshow(img2)
        axs[row, 1].set_title(f"Image {j} (Corr={corr:.2f})")
        axs[row, 1].axis('off')

    plt.suptitle("Most Similar Image Pairs")
    plt.tight_layout()
    plt.show()


# 2️⃣ Embedding Distance Matrix (Feature-level Similarity)
def plot_embedding_distances(X, y, label_names, sample_size=SAMPLE_SIZE):
    sample_faces = []
    sample_labels = []

    for class_idx in range(len(label_names)):
        try:
            idx = y.tolist().index(class_idx)
        except ValueError:
            continue
        face = (X[idx] * 255).astype(np.uint8)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        sample_faces.append(face)
        sample_labels.append(label_names[class_idx])

    embeddings = []
    print("\n🧠 Extracting embeddings...")
    for idx, face in enumerate(sample_faces):
        try:
            emb = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]["embedding"]
            embeddings.append(emb)
        except Exception as e:
            print(f" Failed on sample {idx} - {sample_labels[idx]}: {str(e)}")

    embeddings = np.array(embeddings)
    dist_matrix = cosine_distances(embeddings)

    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, xticklabels=sample_labels, yticklabels=sample_labels,
                cmap='mako', annot=True)
    plt.title("Cosine Distance Between Class Representative Faces (Embedding-Level)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running IID Checks...\n")
    plot_correlation_matrix(X, sample_size=SAMPLE_SIZE)
    plot_embedding_distances(X, y, label_names, sample_size=SAMPLE_SIZE)
