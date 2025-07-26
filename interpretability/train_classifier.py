import pandas as pd  # For better confusion matrix display
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np  # Still using numpy for loading data, but not for KNN core logic

# Load data (assuming these are NumPy arrays from your previous setup)
X_train_np = np.load("X_train_lbp_features.npy")
X_test_np = np.load("X_test_lbp_features.npy")
y_train_np = np.load("y_train.npy")
y_test_np = np.load("y_test.npy")
label_names = np.load("labels.npy")

# Convert NumPy arrays to standard Python lists for manual processing
# This simulates not using NumPy for the core calculations.
X_train = X_train_np.tolist()
X_test = X_test_np.tolist()
y_train = y_train_np.tolist()
y_test = y_test_np.tolist()

# Remove invalid class names
label_names = [name for name in label_names if not name.startswith(
    '.') and not name.endswith('.keep')]

# --- Manual KNN Implementation ---


def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return distance**0.5  # square root


def get_neighbors(X_train, y_train, test_point, k):
    """
    Finds the k nearest neighbors for a given test point.
    Returns a list of (distance, label) tuples for the k neighbors.
    """
    distances = []
    for i in range(len(X_train)):
        train_point = X_train[i]
        label = y_train[i]
        dist = euclidean_distance(test_point, train_point)
        distances.append((dist, label))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get k nearest neighbors
    neighbors = distances[0:k]
    return neighbors


def predict_classification(neighbors):
    """
    Predicts the class label based on the majority vote of neighbors.
    """
    neighbor_labels = [neighbor[1] for neighbor in neighbors]
    # Count occurrences of each label
    label_counts = {}
    for label in neighbor_labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    # Find the label with the maximum count
    # Handle ties by choosing the smallest label or the one encountered first
    # For simplicity, max() with key will pick one if there's a tie
    predicted_label = max(label_counts, key=label_counts.get)
    return predicted_label


def manual_knn(X_train, y_train, X_test, k):
    """
    Performs KNN classification on the test set.
    """
    y_pred_manual = []
    for test_point in X_test:
        neighbors = get_neighbors(X_train, y_train, test_point, k)
        prediction = predict_classification(neighbors)
        y_pred_manual.append(prediction)
    return y_pred_manual


# Set the number of neighbors
k_neighbors = 3

print(f"Manually applying KNN with k={k_neighbors}...")
y_pred_manual = manual_knn(X_train, y_train, X_test, k_neighbors)
print("Manual KNN prediction complete.")

# --- Evaluation (using NumPy for metrics as before, for convenience) ---
# Convert manual predictions back to a NumPy array for sklearn metrics
y_pred = np.array(y_pred_manual)
y_test_eval = y_test_np  # Use the original NumPy y_test for consistent comparison


print("\n--- Evaluation Report (Manual K-Nearest Neighbors) ---")

# Accuracy
accuracy = accuracy_score(y_test_eval, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

# Classification Report (Precision, Recall, F1-Score)
classes = sorted(set(y_test_eval))  # Get unique labels from original y_test
target_names_for_report = [label_names[i] if i <
                           len(label_names) else str(i) for i in classes]
report = classification_report(
    y_test_eval, y_pred, target_names=target_names_for_report, digits=2)
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_eval, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Confusion Matrix with labels
print("\nConfusion Matrix (with labels):")
mapped_label_names = [label_names[i] if i < len(
    label_names) else str(i) for i in classes]
conf_df = pd.DataFrame(
    conf_matrix, index=mapped_label_names, columns=mapped_label_names)
print(conf_df)
