import numpy as np
import cv2
import os
import pandas as pd

# The link is a comment, no action needed for it in the code


def select_features_by_correlation(X_train, correlation_threshold):
    """
    Identifies and removes highly correlated features from a training dataset.

    Args:
        X_train (np.ndarray): The training feature set.
        correlation_threshold (float): The absolute correlation value above which
                                       a feature will be considered redundant.

    Returns:
        list: A list of indices for the features to KEEP.
    """
    print(
        f"\n--- Running Feature Selection (Correlation Threshold = {correlation_threshold}) ---")
    # Create a pandas DataFrame for easy correlation calculation
    df = pd.DataFrame(X_train)

    # Calculate the correlation matrix
    corr_matrix = df.corr().abs()

    # Get the upper triangle of the correlation matrix
    # We don't need to check the whole matrix, as corr(A,B) == corr(B,A)
    # and we don't need to check the diagonal (corr(A,A) == 1)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than the threshold
    to_drop = [column for column in upper_triangle.columns if any(
        upper_triangle[column] > correlation_threshold)]

    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Found {len(to_drop)} highly correlated features to remove.")

    # Create a list of features to keep
    all_feature_indices = list(range(X_train.shape[1]))
    features_to_keep_indices = [
        i for i in all_feature_indices if i not in to_drop]

    print(
        f"Number of features after selection: {len(features_to_keep_indices)}")

    return features_to_keep_indices


# LBP logic


def get_pixel_lbp_value(img, x, y):
    """
    Calculates the LBP value for a single pixel.
    This is the core of the "manual" LBP implementation.
    """
    #  intensity of the center pixel
    center_pixel = img[y, x]

    # Initialize 8-bit binary code for the 8 neighbors to the central pixel
    binary_code = []

    # Array of neighbors of central pixel (Starting NW-N-NE-E-SE-S-SW-WS)
    neighbors = [
        (y-1, x-1), (y-1, x), (y-1, x+1),
        (y,   x+1), (y+1, x+1), (y+1, x),
        (y+1, x-1), (y,   x-1)
    ]

    # Compare center pixel's intensity with its neighbors
    for ny, nx in neighbors:
        if img[ny, nx] >= center_pixel:
            binary_code.append(1)
        else:
            binary_code.append(0)

    # binary to Decimal Conversion
    binary_string = "".join(map(str, binary_code))
    decimal_value = int(binary_string, 2)

    return decimal_value


def calculate_lbp_features(image):
    """
    Takes a single image, calculates the LBP image, and returns its histogram
    which serves as the final feature vector.
    """

    if len(image.shape) == 3 and image.shape[2] == 3:
        # The input is a normalized float color image.
        image_uint8 = (image * 255).astype(np.uint8)
        gray_image = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = (image * 255).astype(np.uint8)

    height, width = gray_image.shape

    # Handle cases where image might be too small for 3x3 LBP window
    if height < 3 or width < 3:
        # Return a zero histogram of expected size (256 for LBP)
        # This prevents errors if a very small face ROI is passed
        return np.zeros(256, dtype=np.float32)

    # Initialize empty LBP img.
    lbp_image = np.zeros((height - 2, width - 2), dtype=np.uint8)

    # Iterate through each pixel
    for y in range(1, height - 1):
        for x in range(1, width - 1):  # Corrected range for x
            lbp_image[y-1, x-1] = get_pixel_lbp_value(gray_image, x, y)

    hist, _ = np.histogram(lbp_image.ravel(),
                           # 256 bins for LBP values 0-255
                           bins=np.arange(0, 257),
                           range=(0, 256))

    # Normalize the histogram
    hist = hist.astype("float")
    # Add a small epsilon to mitigate zero division
    hist /= (hist.sum() + 1e-6)

    return hist


# Processing Script

def main():
    print("Loading preprocessed data...")
    # Assuming these are in the root directory relative to where featureExtraction.py is run
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")

    print(
        f"Loaded {len(X_train)} training images and {len(X_test)} testing images.")

    # Initialize LBP list
    X_train_lbp_features = []
    X_test_lbp_features = []

    print("\nExtracting LBP features from the training set...")
    for i, image in enumerate(X_train):
        features = calculate_lbp_features(image)
        X_train_lbp_features.append(features)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(X_train)} images.")

    print("\nExtracting LBP features from the testing set...")
    for i, image in enumerate(X_test):
        features = calculate_lbp_features(image)
        X_test_lbp_features.append(features)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(X_test)} images.")

    # Converting to np arrays
    X_train_lbp = np.array(X_train_lbp_features)
    X_test_lbp = np.array(X_test_lbp_features)

    # --- APPLY FEATURE SELECTION ---
    # 1. Decide which features to keep based ONLY on the training data
    correlation_threshold = float(
        input("Enter the correlation threshold: "))  # Added colon for clarity
    features_to_keep = select_features_by_correlation(
        X_train_lbp, correlation_threshold)

    # 2. Filter both training and testing sets to keep only those features
    X_train_lbp_selected = X_train_lbp[:, features_to_keep]
    X_test_lbp_selected = X_test_lbp[:, features_to_keep]
    # --- FEATURE SELECTION COMPLETE ---

    # Save the new, reduced feature vectors to disk
    # These will be saved in the CWD of the script execution (i.e., your project root if run from there)
    np.save("X_train_lbp_features.npy", X_train_lbp_selected)
    np.save("X_test_lbp_features.npy", X_test_lbp_selected)
    # <-- THIS IS THE CRUCIAL LINE!
    np.save("features_to_keep.npy", np.array(features_to_keep))

    print("\n Feature Extraction & Selection Complete ")
    print(f"Training features shape: {X_train_lbp_selected.shape}")
    print(f"Testing features shape: {X_test_lbp_selected.shape}")
    print("Saved 'X_train_lbp_features.npy', 'X_test_lbp_features.npy', and 'features_to_keep.npy'.")
    print("Features extracted and saved. Ready for classification.")


# This ensures main() is called only when the script is executed directly
if __name__ == "__main__":
    main()

# Removed the duplicate if __name__ == "__main__": main() at the end
