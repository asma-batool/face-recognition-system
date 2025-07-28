import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
import os
import joblib

# CONFIG
IMG_SIZE = 128
DATASET_PATH = "./dataset"
MODEL_PATH = "./result/custom_knn_model.joblib"

# Custom KNN Implementation


class CustomKNeighborsClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Trains the k-NN classifier by storing the training data.
        Args:
            X (np.array): Training features.
            y (np.array): Training labels.
        """
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        """
        Calculates the Euclidean distance between two data points.
        Args:
            x1 (np.array): First data point.
            x2 (np.array): Second data point.
        Returns:
            float: Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X):
        """
        Predicts the labels for new data points.
        Args:
            X (np.array): New data points to predict.
        Returns:
            np.array: Predicted labels.
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """
        Predicts the label for a single data point.
        Args:
            x (np.array): Single data point to predict.
        Returns:
            Any: Predicted label (most common label among k-nearest neighbors).
        """
        distances = [self._euclidean_distance(
            x, x_train) for x_train in self.X_train]
        # Get the k-nearest neighbors (indices)
        k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
        # Get the labels of the k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        # Return the most common label among the k-nearest neighbors
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]


# --- GUI APP ---
class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition System")
        # Increased window size to accommodate larger video feed
        self.master.geometry("800x700")
        self.master.resizable(False, False)  # Keep fixed size for simplicity

        self.webcam_running = False  # Flag to control webcam thread

        # Main title label
        self.l_main = tk.Label(
            master, text="Face Recognition System", font=("Helvetica", 20, "bold"))
        self.l_main.pack(pady=10)

        # Status label for messages
        self.l_status = tk.Label(
            master, text="", fg="green", font=("Helvetica", 12))
        self.l_status.pack()

        # Button to train the model
        self.b_train = tk.Button(
            master, text="Train Model", width=25, height=2, command=self.train_model)
        self.b_train.pack(pady=10)

        # Button to choose an image for prediction
        self.b_choose = tk.Button(
            master, text="Choose Image for Prediction", width=25, height=2, command=self.choose_image)
        self.b_choose.pack(pady=10)

        # Button to start/stop live webcam detection
        self.b_detect = tk.Button(master, text="Live Webcam Test",
                                  width=25, height=2, command=self.start_webcam_detection)
        self.b_detect.pack(pady=10)

        # Label to display images/webcam feed
        self.label_img = tk.Label(master)
        # Make the image label expand to fill available space
        self.label_img.pack(expand=True, fill=tk.BOTH)

        # Initialize with a black placeholder image
        self.placeholder_img = Image.new('RGB', (640, 480), color='black')
        self.placeholder_tk = ImageTk.PhotoImage(self.placeholder_img)
        self.label_img.config(image=self.placeholder_tk)
        self.label_img.image = self.placeholder_tk  # Keep a reference

    def set_buttons_state(self, state):
        """
        Sets the state of the main control buttons.
        Args:
            state (str): 'normal' to enable, 'disabled' to disable.
        """
        self.b_train.config(state=state)
        self.b_choose.config(state=state)

    def update_status(self, text, color='green'):
        """
        Updates the text and color of the status label.
        Args:
            text (str): The status message.
            color (str): The color of the status message.
        """
        self.l_status.config(text=text, fg=color)

    def preprocess_image(self, img):
        """
        Preprocesses an image for feature extraction.
        Converts to grayscale, resizes, and flattens.
        Args:
            img (np.array): Input image (BGR format).
        Returns:
            np.array: Flattened, preprocessed image vector.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        return resized.flatten()

    def load_dataset(self):
        """
        Loads and preprocesses the dataset from the specified path.
        Returns:
            tuple: (X, y) where X is an array of preprocessed image vectors
                   and y is an array of corresponding labels.
        """
        X, y = [], []
        if not os.path.exists(DATASET_PATH):
            self.update_status(
                f"Dataset path not found: {DATASET_PATH}", 'red')
            return np.array(X), np.array(y)

        for class_label in os.listdir(DATASET_PATH):
            class_dir = os.path.join(DATASET_PATH, class_label)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    # Check if it's a valid image file
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        img = cv2.imread(img_path)
                        if img is not None:
                            X.append(self.preprocess_image(img))
                            y.append(class_label)
        return np.array(X), np.array(y)

    def train_model(self):
        """
        Initiates the model training process in a separate thread.
        """
        self.set_buttons_state('disabled')
        self.update_status("Training model...", '#f39c12')
        # Use a thread to prevent GUI freeze during training
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        """
        Actual model training logic executed in a separate thread.
        """
        try:
            X, y = self.load_dataset()
            if len(X) == 0:
                self.update_status(
                    "No data found in dataset folder. Please ensure it exists and contains images.", 'red')
                return

            # Ensure the result directory exists
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

            # Use custom KNN classifier
            model = CustomKNeighborsClassifier(n_neighbors=3)
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            self.update_status("Model trained and saved!", 'green')
        except Exception as e:
            self.update_status(f"Error during training: {str(e)}", 'red')
        finally:
            self.set_buttons_state('normal')

    def choose_image(self):
        """
        Allows the user to choose an image file for prediction.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        # Stop webcam if it's running
        if self.webcam_running:
            self.webcam_running = False
            self.b_detect.config(text="Live Webcam Test")

        self.update_status("Predicting...", '#f39c12')
        try:
            # Load the trained model
            if not os.path.exists(MODEL_PATH):
                self.update_status(
                    "Model not found. Please train the model first.", 'red')
                return
            model = joblib.load(MODEL_PATH)

            img = cv2.imread(file_path)
            if img is None:
                self.update_status("Failed to load image.", 'red')
                return

            face_vector = self.preprocess_image(img).reshape(1, -1)
            prediction = model.predict(face_vector)[0]
            self.update_status(f"Prediction: {prediction}", 'green')

            # Display the chosen image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize for display in the GUI, keeping a reasonable size
            img_pil = Image.fromarray(img_rgb)
            # Use LANCZOS for better quality downsampling
            img_pil = img_pil.resize((640, 480), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.label_img.config(image=img_tk)
            self.label_img.image = img_tk  # Keep a reference to prevent garbage collection
        except Exception as e:
            self.update_status(f"Error during prediction: {str(e)}", 'red')

    def start_webcam_detection(self):
        """
        Starts or stops the live webcam detection.
        """
        if not self.webcam_running:
            self.webcam_running = True
            self.set_buttons_state('disabled')
            self.update_status("Starting webcam...", '#3498db')
            self.b_detect.config(text="Stop Webcam")
            # Start webcam detection in a separate thread
            threading.Thread(
                target=self._webcam_detection_thread, daemon=True).start()
        else:
            self.webcam_running = False
            self.b_detect.config(text="Live Webcam Test")
            self.update_status("Webcam stopped.", '#8e44ad')
            self.set_buttons_state('normal')

    def _webcam_detection_thread(self):
        """
        Handles the webcam feed processing and prediction in a separate thread.
        """
        try:
            # Load the trained model
            if not os.path.exists(MODEL_PATH):
                self.update_status(
                    "Model not found. Please train the model first.", 'red')
                self.set_buttons_state('normal')
                return
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}", 'red')
            self.set_buttons_state('normal')
            self.webcam_running = False  # Ensure flag is reset on error
            return

        cap = cv2.VideoCapture(0)  # Open default camera
        if not cap.isOpened():
            self.update_status(
                "Failed to open webcam. Make sure it's connected and not in use.", 'red')
            self.set_buttons_state('normal')
            self.webcam_running = False  # Ensure flag is reset on error
            return

        try:
            while self.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    self.update_status(
                        "Failed to read frame from webcam.", 'red')
                    break

                # Preprocess the frame for prediction
                face_vector = self.preprocess_image(frame).reshape(1, -1)
                prediction = model.predict(face_vector)[0]

                # Draw prediction text on the frame
                cv2.putText(frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                # Convert frame to PhotoImage for Tkinter display
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                # Resize to a larger, fixed size for consistent display
                img_pil = img_pil.resize((640, 480), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img_pil)

                # Update the Tkinter label with the new frame
                self.label_img.config(image=img_tk)
                self.label_img.image = img_tk  # Keep a reference

                # Small delay to prevent burning CPU (adjust as needed)
                # This is important for smooth GUI updates
                self.master.update_idletasks()
                self.master.update()

        except Exception as e:
            self.update_status(f"Error during webcam feed: {str(e)}", 'red')
        finally:
            cap.release()  # Release the webcam resource
            self.webcam_running = False  # Reset flag
            self.b_detect.config(text="Live Webcam Test")
            self.set_buttons_state('normal')
            # Restore placeholder image after webcam stops
            self.label_img.config(image=self.placeholder_tk)
            self.label_img.image = self.placeholder_tk


# --- MAIN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
