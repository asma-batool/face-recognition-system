

import numpy as np
import cv2
import os
import itertools
import random
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
import warnings

# Suppress runtime warnings for cleaner output
warnings.filterwarnings("ignore")

# --- LBP Feature Extraction Functions ---
# This section makes the script self-contained by including the necessary
# feature extraction logic directly.

def get_pixel_lbp_value(img, x, y):
    """
    Calculates the LBP value for a single pixel.
    """
    center_pixel = img[y, x]
    binary_code = []
    # Clockwise neighbors starting from top-left
    neighbors = [
        (y-1, x-1), (y-1, x), (y-1, x+1),
        (y,   x+1), (y+1, x+1), (y+1, x),
        (y+1, x-1), (y,   x-1)
    ]
    for ny, nx in neighbors:
        if img[ny, nx] >= center_pixel:
            binary_code.append(1)
        else:
            binary_code.append(0)
    binary_string = "".join(map(str, binary_code))
    decimal_value = int(binary_string, 2)
    return decimal_value

def calculate_lbp_features(image):
    """
    Takes an image, calculates the LBP image, and returns its histogram
    which serves as the final feature vector.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Input is a normalized float color image (0-1). Convert to uint8 (0-255).
        image_uint8 = (image * 255).astype(np.uint8)
        gray_image = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    else:
        # Assume it's already a grayscale image
        gray_image = image

    height, width = gray_image.shape
    # LBP image will be smaller as we can't process borders
    lbp_image = np.zeros((height - 2, width - 2), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            lbp_image[y-1, x-1] = get_pixel_lbp_value(gray_image, x, y)

    # Calculate histogram of the LBP image
    hist, _ = np.histogram(lbp_image.ravel(),
                           bins=np.arange(0, 257),
                           range=(0, 256))
    # Normalize histogram to create a feature vector
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# --- Analysis Functions ---

def plot_correlation_matrix(X: np.ndarray, sample_size: int) -> None:
    """Computes and plots correlation between image pixels across a sample."""
    print("Plotting pixel-level correlation matrix for a random sample...")
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

    print("\nTop 5 Most Correlated Image Pairs (from sample):")
    for i, j, corr in top_pairs:
        print(f"Image {i} and {j} -- Correlation: {corr:.4f}")

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

def plot_feature_vector_distances(X, y, label_names):
    """Extracts LBP features for each class and visualizes class-wise cosine distances."""
    print("\nExtracting LBP features to compute mean class representatives...")
    class_images = defaultdict(list)
    for i, label in enumerate(y):
        class_images[label].append(X[i])

    mean_feature_vectors = []
    class_labels_ordered = []

    for label_idx, images in sorted(class_images.items()):
        class_name = label_names[label_idx]
        print(f"  Processing class '{class_name}' ({len(images)} images)...")
        
        # Generate LBP feature vectors for all images in the class
        feature_vectors = [calculate_lbp_features(img) for img in images]
        
        if feature_vectors:
            # Calculate the mean feature vector for the class
            mean_vec = np.mean(feature_vectors, axis=0)
            mean_feature_vectors.append(mean_vec)
            class_labels_ordered.append(class_name)

    if not mean_feature_vectors:
        print("No feature vectors were generated. Skipping plot.")
        return

    # Compute the pairwise cosine distance between all mean class vectors
    dist_matrix = cosine_distances(np.array(mean_feature_vectors))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, xticklabels=class_labels_ordered, yticklabels=class_labels_ordered,
                cmap='mako_r', annot=True, fmt=".2f")
    plt.title("Cosine Distance Between Mean Class LBP Feature Vectors")
    plt.tight_layout()
    plt.show()

def plot_class_variance(X, y, label_names):
    """Plots variance within each class to highlight noisier categories."""
    print("\nCalculating class-wise image variance...")
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
    print("\nDetecting possible duplicate images...")
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
        print("No significant duplicates found.")
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

def remove_similar_samples_per_class(X, y, threshold=0.99, remove_per_class=10):
    """
    Removes a few highly similar (correlated) images from each class.
    Returns new arrays: X_new, y_new
    """
    print("\nRemoving near-duplicate images per class...")
    keep_indices = []
    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        if len(idxs) <= 1:
            keep_indices.extend(idxs)
            continue
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
    print(f"Removed {len(X) - len(keep_indices)} samples from dataset.")
    return X[keep_indices], y[keep_indices]

# --- Main Execution ---
if __name__ == "__main__":
    # Configuration
    SAMPLE_SIZE = 150
    
    # Load data
    try:
        X = np.load("X_train.npy")
        y = np.load("y_train.npy")
        label_names = np.load("labels.npy", allow_pickle=True)
    except FileNotFoundError:
        print("Error: Make sure 'X_train.npy', 'y_train.npy', and 'labels.npy' are present.")
        print("Run the preprocessing script first.")
        exit()
        
    print("Running Dataset Sanity Checks...\n" + "="*40)

    # Step 0: Optional cleaning
    X, y = remove_similar_samples_per_class(X, y, threshold=0.995, remove_per_class=3)
    print("="*40)

    # Step 1: Pixel-level similarity
    plot_correlation_matrix(X, sample_size=min(SAMPLE_SIZE, len(X)))
    print("="*40)

    # Step 2: LBP Feature-based class distance
    plot_feature_vector_distances(X, y, label_names)
    print("="*40)

    # Step 3: Class-wise variance
    plot_class_variance(X, y, label_names)
    print("="*40)

    # Step 4: Duplicate visualization (final check)
    find_duplicate_images(X, threshold=0.99)
    print("\nAll checks complete.")
