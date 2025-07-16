# Suppress TensorFlow and other warnings for clean output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

# Essential libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_distances
import itertools
from collections import defaultdict

# ------------------ Configuration ------------------
X = np.load("X_train.npy")              # shape: (num_samples, height, width, channels)
y = np.load("y_train.npy")              # class index for each image
label_names = np.load("labels.npy", allow_pickle=True)  # maps class index to class name

SAMPLE_SIZE = 150
FACE_RECOGNITION_MODEL = 'Facenet'

# ------------------ Analysis Functions ------------------

def plot_correlation_matrix(X: np.ndarray, sample_size: int = SAMPLE_SIZE) -> None:
    """Computes and plots correlation between image pixels across a sample."""
    print("📊 Plotting pixel-level correlation matrix for a random sample...")
    indices = random.sample(range(len(X)), sample_size)
    sample_images = X[indices].reshape(sample_size, -1)
    corr_matrix = np.corrcoef(sample_images)

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, cmap='coolwarm', cbar_kws={'label': 'Correlation'}, square=True)
    plt.title("Image Correlation Matrix (Pixel-Level)")
    plt.tight_layout()
    plt.show()

    pairs = list(itertools.combinations(range(sample_size), 2))
    correlations = [(i, j, corr_matrix[i, j]) for (i, j) in pairs]
    top_pairs = sorted(correlations, key=lambda x: -abs(x[2]))[:5]

    print("\n🔍 Top 5 Most Correlated Image Pairs (from sample):")
    for i, j, corr in top_pairs:
        print(f"Image {i} and {j} — Correlation: {corr:.4f}")

    fig, axs = plt.subplots(len(top_pairs), 2, figsize=(6, 10))
    fig.suptitle("Most Similar Image Pairs (Pixel-Level)", fontsize=16)
    for row, (i, j, corr) in enumerate(top_pairs):
        img1 = (X[indices[i]] * 255).astype(np.uint8)
        img2 = (X[indices[j]] * 255).astype(np.uint8)
        axs[row, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axs[row, 0].set_title(f"Image {i}")
        axs[row, 0].axis('off')
        axs[row, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axs[row, 1].set_title(f"Image {j} (Corr={corr:.2f})")
        axs[row, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_embedding_distances_robust(X, y, label_names, model_name):
    """Extracts embeddings for each class and visualizes class-wise cosine distances."""
    print("\n🧠 Extracting embeddings to compute mean class representatives...")
    class_images = defaultdict(list)
    for i, label in enumerate(y):
        img = (X[i] * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_images[label].append(img)

    mean_embeddings = []
    class_labels_ordered = []

    for label_idx, images in sorted(class_images.items()):
        class_name = label_names[label_idx]
        print(f"  Processing class '{class_name}' ({len(images)} images)...")
        embeddings = []
        for img in images:
            try:
                result = DeepFace.represent(img_path=img, model_name=model_name, enforce_detection=False)
                embeddings.append(result[0]['embedding'])
            except Exception as e:
                print(f"     ❌ Error on one image in class {class_name}: {e}")
        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            mean_embeddings.append(mean_emb)
            class_labels_ordered.append(class_name)

    if not mean_embeddings:
        print("❌ No embeddings generated. Skipping plot.")
        return

    dist_matrix = cosine_distances(np.array(mean_embeddings))
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, xticklabels=class_labels_ordered, yticklabels=class_labels_ordered,
                cmap='mako_r', annot=True, fmt=".2f")
    plt.title(f"Cosine Distance Between Mean Class Embeddings ({model_name})")
    plt.tight_layout()
    plt.show()

def plot_class_variance(X, y, label_names):
    """Plots variance within each class to highlight noisier categories."""
    print("\n📈 Calculating class-wise image variance...")
    variances = defaultdict(float)
    for label in np.unique(y):
        class_images = X[y == label]
        variances[label_names[label]] = np.var(class_images)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(variances.keys()), y=list(variances.values()), palette='viridis')
    plt.xticks(rotation=45, ha="right")
    plt.title("Class-wise Image Variance")
    plt.ylabel("Variance")
    plt.xlabel("Class Label")
    plt.tight_layout()
    plt.show()

def find_duplicate_images(X, threshold=0.99, max_results=5):
    """Visualizes potential near-duplicate image pairs based on correlation."""
    print("\n🔍 Detecting possible duplicate images...")
    flattened = X.reshape(X.shape[0], -1)
    corr_matrix = np.corrcoef(flattened)
    np.fill_diagonal(corr_matrix, 0)
    duplicate_pairs = np.argwhere(corr_matrix >= threshold)
    seen = set()
    duplicates_to_show = []
    for i, j in duplicate_pairs:
        if i != j and tuple(sorted((i, j))) not in seen:
            score = corr_matrix[i, j]
            duplicates_to_show.append((i, j, score))
            seen.add(tuple(sorted((i, j))))
            if len(duplicates_to_show) >= max_results:
                break
    if not duplicates_to_show:
        print("✅ No significant duplicates found.")
        return
    for i, j, score in duplicates_to_show:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        fig.suptitle(f"Potential Duplicates (Corr={score:.3f})", fontsize=14)
        axs[0].imshow(cv2.cvtColor((X[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Image Index: {i}")
        axs[0].axis('off')
        axs[1].imshow(cv2.cvtColor((X[j] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Image Index: {j}")
        axs[1].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

# 🔹 5. Auto-Remove Similar Images Within Classes
def remove_similar_samples_per_class(X, y, threshold=0.99, remove_per_class=10):
    """
    Removes a few highly similar (correlated) images from each class.
    Returns new arrays: X_new, y_new
    """
    print("\n🧹 Removing near-duplicate images per class...")
    keep_indices = []
    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        class_images = X[idxs].reshape(len(idxs), -1)
        corr_matrix = np.corrcoef(class_images)
        np.fill_diagonal(corr_matrix, 0)
        similar_pairs = np.argwhere(corr_matrix >= threshold)
        removed = set()
        for i, j in similar_pairs:
            if len(removed) >= remove_per_class:
                break
            # Remove one of the pair (not both)
            to_remove = idxs[j]
            removed.add(to_remove)
        keep_indices.extend([i for i in idxs if i not in removed])
    print(f"➡️ Removed {len(X) - len(keep_indices)} samples from dataset.")
    return X[keep_indices], y[keep_indices]

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    print("🚀 Running Dataset Sanity Checks...\n" + "="*40)

    # Step 0: Optional cleaning
    X, y = remove_similar_samples_per_class(X, y, threshold=0.995, remove_per_class=3)
    print("="*40)

    # Step 1: Pixel-level similarity
    plot_correlation_matrix(X, sample_size=SAMPLE_SIZE)
    print("="*40)

    # Step 2: Embedding-based class distance
    plot_embedding_distances_robust(X, y, label_names, model_name=FACE_RECOGNITION_MODEL)
    print("="*40)

    # Step 3: Class-wise variance
    plot_class_variance(X, y, label_names)
    print("="*40)

    # Step 4: Duplicate visualization (final check)
    find_duplicate_images(X, threshold=0.99)
    print("\n✅ All checks complete.")
