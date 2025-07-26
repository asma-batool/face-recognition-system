import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import joblib

# CONFIG
IMG_SIZE = 128
DATASET_PATH = "./dataset"
MODEL_PATH = "./result/knn_model.joblib"

# --- GUI APP ---
class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition System")
        self.master.geometry("600x450")
        self.master.resizable(False, False)
        self.webcam_running = False  # flag to stop webcam

        self.l_main = tk.Label(master, text="Face Recognition System", font=("Helvetica", 20, "bold"))
        self.l_main.pack(pady=10)

        self.l_status = tk.Label(master, text="", fg="green", font=("Helvetica", 12))
        self.l_status.pack()

        self.b_train = tk.Button(master, text="Train Model", width=25, height=2, command=self.train_model)
        self.b_train.pack(pady=10)

        self.b_choose = tk.Button(master, text="Choose Image for Prediction", width=25, height=2, command=self.choose_image)
        self.b_choose.pack(pady=10)

        self.b_detect = tk.Button(master, text="Live Webcam Test", width=25, height=2, command=self.start_webcam_detection)
        self.b_detect.pack(pady=10)

        self.label_img = tk.Label(master)
        self.label_img.pack()

    def set_buttons_state(self, state):
        self.b_train.config(state=state)
        self.b_choose.config(state=state)

    def update_status(self, text, color='green'):
        self.l_status.config(text=text, fg=color)

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        return resized.flatten()

    def load_dataset(self):
        X, y = [], []
        for class_label in os.listdir(DATASET_PATH):
            class_dir = os.path.join(DATASET_PATH, class_label)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        X.append(self.preprocess_image(img))
                        y.append(class_label)
        return np.array(X), np.array(y)

    def train_model(self):
        self.set_buttons_state('disabled')
        self.update_status("Training model...", '#f39c12')
        try:
            X, y = self.load_dataset()
            if len(X) == 0:
                self.update_status("No data found in dataset folder.", 'red')
                return
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            self.update_status("Model trained and saved!", 'green')
        except Exception as e:
            self.update_status(f"Error: {str(e)}", 'red')
        finally:
            self.set_buttons_state('normal')

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        self.update_status("Predicting...", '#f39c12')
        try:
            model = joblib.load(MODEL_PATH)
            img = cv2.imread(file_path)
            face_vector = self.preprocess_image(img).reshape(1, -1)
            prediction = model.predict(face_vector)[0]
            self.update_status(f"Prediction: {prediction}", 'green')

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img_pil)
            self.label_img.config(image=img_tk)
            self.label_img.image = img_tk
        except Exception as e:
            self.update_status(f"Error: {str(e)}", 'red')

    def start_webcam_detection(self):
        if not self.webcam_running:
            self.webcam_running = True
            self.set_buttons_state('disabled')
            self.update_status("Starting webcam...", '#3498db')
            self.b_detect.config(text="Stop Webcam")
            threading.Thread(target=self._webcam_detection_thread, daemon=True).start()
        else:
            self.webcam_running = False
            self.b_detect.config(text="Live Webcam Test")
            self.update_status("Webcam stopped.", '#8e44ad')
            self.set_buttons_state('normal')

    def _webcam_detection_thread(self):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}", 'red')
            self.set_buttons_state('normal')
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.update_status("Failed to open webcam.", 'red')
            self.set_buttons_state('normal')
            return

        try:
            while self.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    break

                face_vector = self.preprocess_image(frame).reshape(1, -1)
                prediction = model.predict(face_vector)[0]

                cv2.putText(frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil = img_pil.resize((300, 300))
                img_tk = ImageTk.PhotoImage(img_pil)
                self.label_img.config(image=img_tk)
                self.label_img.image = img_tk
        finally:
            cap.release()
            self.webcam_running = False
            self.b_detect.config(text="Live Webcam Test")
            self.set_buttons_state('normal')

# --- MAIN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()