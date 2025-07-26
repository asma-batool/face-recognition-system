from collections import defaultdict
import itertools
import cv2
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

# Configuration
X = np.load("X_train.npy")
y = np.load("y_train.npy")              # class index for each image
label_names = np.load("labels.npy", allow_pickle=True)

SAMPLE_SIZE = 150

#  Analysis Functions


def plot_correlation_matrix(X: np.ndarray, sample_size: int = SAMPLE_SIZE, title="Image Correlation Matrix (Pixel-Level)") -> None:
    """Computes and plots correlation between image pixels across a sample."""
    print(
        f"Plotting pixel-level correlation matrix for a random sample... ({title})")
    indices = random.sample(range(len(X)), sample_size)
    sample_images = X[indices].reshape(sample_size, -1)
    corr_matrix = np.corrcoef(sample_images)

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, cmap='coolwarm', cbar_kws={
                'label': 'Correlation'}, square=True)
    plt.title(title)
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


def plot_class_variance(X, y, label_names):
    """Plots variance within each class to highlight noisier categories."""
    print("\n Calculating class-wise image variance...")
    variances = defaultdict(float)
    for label in np.unique(y):
        class_images = X[y == label]
        variances[label_names[label]] = np.var(class_images)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(variances.keys()), y=list(
        variances.values()), palette='viridis')
    plt.xticks(rotation=45, ha="right")
    plt.title("Class-wise Image Variance")
    plt.ylabel("Variance")
    plt.xlabel("Class Label")
    plt.tight_layout()
    plt.show()


def find_duplicate_images(X, threshold=0.99, max_results=5):
    """Visualizes potential near-duplicate image pairs based on correlation."""
    print("\n Detecting possible duplicate images...")
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
        print(" No significant duplicates found.")
        return
    for i, j, score in duplicates_to_show:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        fig.suptitle(f"Potential Duplicates (Corr={score:.3f})", fontsize=14)
        axs[0].imshow(cv2.cvtColor(
            (X[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Image Index: {i}")
        axs[0].axis('off')
        axs[1].imshow(cv2.cvtColor(
            (X[j] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Image Index: {j}")
        axs[1].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()


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
    print(f" Removed {len(X) - len(keep_indices)} samples from dataset.")
    return X[keep_indices], y[keep_indices]


def show_class_outliers(X, y, label_names, top_n=10):
    import matplotlib.pyplot as plt
    for label in np.unique(y):
        class_indices = np.where(y == label)[0]
        class_images = X[class_indices]
        class_mean = np.mean(class_images, axis=0)
        # Compute L2 distance from mean for each image
        distances = np.linalg.norm(class_images - class_mean, axis=(1, 2))
        outlier_indices = np.argsort(-distances)[:top_n]
        print(f"\nClass {label_names[label]}: Top {top_n} outliers")
        plt.figure(figsize=(top_n*2, 2))
        for i, idx in enumerate(outlier_indices):
            plt.subplot(1, top_n, i+1)
            plt.imshow(class_images[idx], cmap='gray')
            plt.title(f"X idx: {class_indices[idx]}")
            plt.axis('off')
        plt.show()


def plot_class_counts(y, label_names):
    import matplotlib.pyplot as plt
    from collections import Counter
    counts = Counter(y)
    plt.figure(figsize=(8, 4))
    plt.bar([label_names[i] for i in counts.keys()], counts.values())
    plt.title("Number of Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


def show_random_samples(X, y, label_names, samples_per_class=5):
    import matplotlib.pyplot as plt
    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        chosen = np.random.choice(
            idxs, min(samples_per_class, len(idxs)), replace=False)
        plt.figure(figsize=(samples_per_class*2, 2))
        plt.suptitle(f"Random samples from class {label_names[label]}")
        for i, idx in enumerate(chosen):
            plt.subplot(1, samples_per_class, i+1)
            plt.imshow(X[idx], cmap='gray')
            plt.axis('off')
        plt.show()


# Main Execution
if __name__ == "__main__":
    print(" Running Dataset Sanity Checks...\n" + "="*40)

    # Step 0: Show BEFORE cleaning
    plot_correlation_matrix(X, sample_size=SAMPLE_SIZE,
                            title="Correlation Matrix BEFORE Duplicate Removal")
    print("="*40)

    # Step 1: Optional cleaning
    X, y = remove_similar_samples_per_class(
        X, y, threshold=0.995, remove_per_class=80)
    print("="*40)

    # Step 2: Show AFTER cleaning
    plot_correlation_matrix(X, sample_size=SAMPLE_SIZE,
                            title="Correlation Matrix AFTER Duplicate Removal")
    print("="*40)

    plot_class_counts(y, label_names)
    show_random_samples(X, y, label_names)

    show_class_outliers(X, y, label_names, top_n=5)

    # Interactive manual removal of outlier indices
    print("\n📝 Enter image indices to remove (comma-separated). Press 'q' or Enter to finish.")

    manual_remove_indices = set()

    while True:
        user_input = input(
            "Indices to remove (e.g., 12,45,78) or 'q' to quit: ").strip()
        if user_input.lower() in ["q", ""]:
            break
        try:
            new_indices = [int(idx.strip()) for idx in user_input.split(
                ",") if idx.strip().isdigit()]
            manual_remove_indices.update(new_indices)
            print(
                f"✅ Added {len(new_indices)} indices. Total to remove: {len(manual_remove_indices)}")
        except Exception as e:
            print(
                f"⚠️ Invalid input. Please enter comma-separated numbers only. ({e})")

    # Apply removal
    if manual_remove_indices:
        mask = np.ones(len(X), dtype=bool)
        mask[list(manual_remove_indices)] = False
        X_clean = X[mask]
        y_clean = y[mask]
        print(f"\n🧹 Removed {len(manual_remove_indices)} manual indices.")
    else:
        X_clean = X
        y_clean = y
        print("\n🚫 No manual indices were removed.")

    show_class_outliers(X, y, label_names, top_n=5)

    # Step 2: (Embedding-based class distance removed, not allowed)
    # plot_embedding_distances_robust(X, y, label_names, model_name=FACE_RECOGNITION_MODEL)
    # print("="*40)

    # Use cleaned data for further analysis
    plot_class_variance(X_clean, y_clean, label_names)
    print("="*40)

    find_duplicate_images(X_clean, threshold=0.99)
    print("\n✅ All checks complete.")

    # Save cleaned arrays (overwrite original ones if you want permanent effect)
    np.save("X_train.npy", X_clean)
    np.save("y_train.npy", y_clean)
