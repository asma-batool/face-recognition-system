from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load data
X_train = np.load("X_train_lbp_features.npy")
X_test = np.load("X_test_lbp_features.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
label_names = np.load("labels.npy")

# Remove invalid class names
label_names = [name for name in label_names if not name.startswith(
    '.') and not name.endswith('.keep')]

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Get unique labels in y_test
classes = sorted(set(y_test))
class_to_index = {cls: i for i, cls in enumerate(classes)}
num_classes = len(classes)

# Initialize confusion matrix
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

# Fill confusion matrix
for true_label, pred_label in zip(y_test, y_pred):
    i = class_to_index[true_label]
    j = class_to_index[pred_label]
    conf_matrix[i][j] += 1

# Accuracy
correct_preds = np.trace(conf_matrix)
total_preds = np.sum(conf_matrix)
accuracy = correct_preds / total_preds

# Classification Report (Precision, Recall, F1)
print("\n--- Evaluation Report ---")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Class\tPrec\tRec\tF1\tSupport")

for i, cls in enumerate(classes):
    TP = conf_matrix[i][i]
    FP = np.sum(conf_matrix[:, i]) - TP
    FN = np.sum(conf_matrix[i, :]) - TP
    support = np.sum(conf_matrix[i, :])

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) else 0

    label = label_names[cls] if cls < len(label_names) else str(cls)
    print(f"{label}\t{precision:.2f}\t{recall:.2f}\t{f1:.2f}\t{support}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(conf_matrix)
