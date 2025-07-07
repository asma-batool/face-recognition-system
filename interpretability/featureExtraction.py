import numpy as np
import cv2
import os
import numpy as np
import cv2

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

    # Initialize empty LBP img.
    lbp_image = np.zeros((height - 2, width - 2), dtype=np.uint8)

    # Iterate through each pixel
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            lbp_image[y-1, x-1] = get_pixel_lbp_value(gray_image, x, y)

    hist, _ = np.histogram(lbp_image.ravel(),
                           bins=np.arange(0, 257),
                           range=(0, 256))

    # Normalize the histogram
    hist = hist.astype("float")
    # Add a small epsilon to mitigate zero devision
    hist /= (hist.sum() + 1e-6)

    return hist


# Processing Script

def main():
    print("Loading preprocessed data...")
    # Load train data
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")

    print(
        f"Loaded {len(X_train)} training images and {len(X_test)} testing images.")

    # Initialize LBP list
    X_train_lbp_features = []
    X_test_lbp_features = []

    print("\nExtracting LBP features from the training set...")
    for i, image in enumerate(X_train):
        # Calculate the LBP feature vector for each training image
        features = calculate_lbp_features(image)
        X_train_lbp_features.append(features)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(X_train)} images.")

    print("\nExtracting LBP features from the testing set...")
    for i, image in enumerate(X_test):
        # Calculate the LBP feature vector for each testing image
        features = calculate_lbp_features(image)
        X_test_lbp_features.append(features)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(X_test)} images.")

    # Converting to np arrays
    X_train_lbp = np.array(X_train_lbp_features)
    X_test_lbp = np.array(X_test_lbp_features)

    # Save the new feature vectors to disk
    np.save("X_train_lbp_features.npy", X_train_lbp)
    np.save("X_test_lbp_features.npy", X_test_lbp)

    print("\n Feature Extraction Complete ")
    print(f"Training features shape: {X_train_lbp.shape}")
    print(f"Testing features shape: {X_test_lbp.shape}")
    print("Saved 'X_train_lbp_features.npy' and 'X_test_lbp_features.npy'.")
    print("Features extracted and saved. Ready for classification.")


if __name__ == "__main__":
    main()
