from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Load data
X_train = np.load("X_train_lbp_features.npy")
X_test = np.load("X_test_lbp_features.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
label_names = np.load("labels.npy")

# Remove bad class names like ".DS_Store" or "dataset.keep"
label_names = [name for name in label_names if not name.startswith(
    '.') and not name.endswith('.keep')]

# Train classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Re-map label names to only present classes
used_label_indices = sorted(set(y_test))
filtered_label_names = [label_names[i] for i in used_label_indices]

# Report
print("\n--- Evaluation Report ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred,
      labels=used_label_indices, target_names=filtered_label_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
