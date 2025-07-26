import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# === Configuration ===
DATASET_PATH = "dataset/"
IMG_SIZE = 128
X = []  # will store the **processed face images**
y = []  # will store the **corresponding labels**

# === Utility Functions ===


def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(img, face_cascade, counter):
    # Resize if image is too large
    max_dim = 400
    if img.shape[0] > max_dim or img.shape[1] > max_dim:
        scale = max_dim / max(img.shape[0], img.shape[1])
        img = cv2.resize(
            img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    eyes_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml")

    # Try multiple cascades
    cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades +
                              "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades +
                              "haarcascade_frontalface_alt2.xml"),
    ]
    faces = []
    for cascade in cascades:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3)
        if len(faces) > 0:
            break
    # Fallback to profile face (left and right)
    if len(faces) == 0:
        profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml")
        faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3)
        if len(faces) == 0:
            profile_cascade_r = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_profileface.xml")
            flipped = cv2.flip(gray, 1)
            faces = profile_cascade_r.detectMultiScale(
                flipped, scaleFactor=1.05, minNeighbors=3)

    if len(faces) == 0:
        counter["undetected"] += 1
        return None

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(face_gray)
        if len(eyes) >= 2:
            cx = x + w // 2
            cy = y + h // 2
            crop_size = max(w, h)
            crop_size = np.clip(crop_size, 80, min(
                img.shape[0], img.shape[1], 220))
            crop_x1 = max(0, cx - crop_size // 2)
            crop_y1 = max(0, cy - crop_size // 2)
            crop_x2 = min(img.shape[1], crop_x1 + crop_size)
            crop_y2 = min(img.shape[0], crop_y1 + crop_size)
            face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            return face_gray
        else:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            return face_gray

    return None


# === Dataset Collection ===

def generate_dataset(user_id, num_samples=200, save_dir=DATASET_PATH):
    save_path = os.path.join(save_dir, str(user_id))
    os.makedirs(save_path, exist_ok=True)

    face_cascade = get_face_cascade()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    print("📸 Capturing images. Press 'q' to quit.")
    img_id = 0

    while img_id < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam.")
            break

        undetected_counter = {"undetected": 0}
        face = crop_face(frame, face_cascade, undetected_counter)
        if face is not None:
            # Convert to grayscale
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(
                save_path, f"{img_id+1}.jpg"), face_resized)

            # For display, convert grayscale back to BGR so imshow works
            display_img = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
            cv2.putText(display_img, str(img_id+1), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face", display_img)

            img_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Dataset collection complete.")


# === Preprocessing ===

def preprocess_dataset(data_path=DATASET_PATH, img_size=IMG_SIZE):
    X, y = [], []
    face_cascade = get_face_cascade()

    label_names = sorted([
        name for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name)) and name != "test"])

    class_frame_counts = {}         # Total frames per class
    class_undetected_counts = {}    # Undetected frames per class
    total_undetected = 0

    for label_index, folder in enumerate(label_names):
        folder_path = os.path.join(data_path, folder)
        print(f"📂 Processing folder: {folder_path}")

        files = sorted([f for f in os.listdir(folder_path)
                       if f.lower().endswith(".jpg")])
        class_frame_counts[folder] = len(files)
        class_undetected_counts[folder] = 0

        for file in files:
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Failed to read {img_path}")
                continue

            undetected_counter = {"undetected": 0}
            face = crop_face(img, face_cascade, undetected_counter)
            if face is None or face.size == 0:
                print(f"❌ No face found in image: {img_path}")
                class_undetected_counts[folder] += 1
                total_undetected += 1
                continue
            face_resized = cv2.resize(face, (img_size, img_size)) / 255.0
            X.append(face_resized)
            y.append(label_index)
            print(f"✅ Face found in: {img_path}")

    if not X:
        print("❌ No valid face data found.")
        return

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("labels.npy", label_names)

    print("\n✅ Preprocessing complete.")
    print(f"Total images: {len(X)}")
    print(f"📊 Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Labels: {label_names}")
    print("\nFrames per class and undetectable frames:")
    for folder in label_names:
        print(
            f"  - {folder}: {class_frame_counts[folder]} frames, {class_undetected_counts[folder]} undetectable")
    print(f"\nTotal undetected faces: {total_undetected}")


# === Test Image Collection ===

def collect_test_images(num_samples=20, save_dir=DATASET_PATH):
    test_path = os.path.join(save_dir, "test")
    os.makedirs(test_path, exist_ok=True)

    face_cascade = get_face_cascade()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        return

    print("🎥 Collecting test images. Press 'q' to quit.")
    img_id = 0

    while img_id < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("❌ Webcam read failed.")
            break

        face = crop_face(frame, face_cascade)
        if face is not None:
            img_id += 1
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            file_path = os.path.join(test_path, f"test_{img_id}.jpg")
            cv2.imwrite(file_path, face_resized)

            cv2.putText(face_resized, str(img_id), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Test Image", face_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test image collection finished.")


def show_sample_images_per_class(X, y, label_names, samples_per_class=10):
    import matplotlib.pyplot as plt
    shown = set()
    plt.figure(figsize=(samples_per_class * 2, len(label_names) * 2))
    for label_idx, class_name in enumerate(label_names):
        class_indices = np.where(y == label_idx)[0]
        for i, idx in enumerate(class_indices[:samples_per_class]):
            plt.subplot(len(label_names), samples_per_class,
                        label_idx * samples_per_class + i + 1)
            img = X[idx]
            if img.ndim == 2:  # grayscale
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.axis('off')
            if i == 0:
                plt.title(f"{class_name}")
    plt.suptitle("Sample cropped grayscale images per class")
    plt.tight_layout()
    plt.show()

# === Main Menu ===


def main():
    print("\n📌 Face Recognition System")
    print("1️⃣  Train from dataset (preprocess existing data)")
    print("2️⃣  Collect test images from webcam only")
    print("3️⃣  Collect test images + Train model")
    print("4️⃣  Collect new dataset images from webcam")

    choice = input("\nSelect an option (1, 2, 3, or 4): ").strip()

    if choice == "1":
        preprocess_dataset()
    elif choice == "2":
        collect_test_images(num_samples=150)
    elif choice == "3":
        preprocess_dataset()
        collect_test_images(num_samples=150)
    elif choice == "4":
        user_id = input("Enter user ID (number only): ").strip()
        if not user_id.isdigit():
            print("❌ Invalid user ID. Use digits only.")
            return
        generate_dataset(user_id=user_id, num_samples=200)
    else:
        print("❌ Invalid option. Exiting.")
        print(" Done preprocessing with face cropping.")


if __name__ == "__main__":
    main()
    # After preprocessing, show sample images for verification
    if os.path.exists("X_train.npy") and os.path.exists("y_train.npy") and os.path.exists("labels.npy"):
        X = np.load("X_train.npy")
        y = np.load("y_train.npy")
        label_names = np.load("labels.npy", allow_pickle=True)
        show_sample_images_per_class(X, y, label_names, samples_per_class=2)
