import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import threading
import time
import PIL.Image, PIL.ImageTk

class SignLanguageRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition System")
        self.root.geometry("1200x800")

        # Load Model
        self.model, self.class_indices = self.load_recognition_model()
        
        # Create UI Components
        self.create_ui()

        # Camera variables
        self.camera_active = False
        self.capture = None
        self.prediction_thread = None

    def load_recognition_model(self):
        try:
            model = load_model('models/final_model.h5')
            with open('models/class_indices.json', 'r') as f:
                class_indices = json.load(f)
            print("Model loaded successfully")
            return model, {v: k for k, v in class_indices.items()}
        except Exception as e:
            messagebox.showerror("Model Loading Error", str(e))
            return None, None

    def create_ui(self):
        # Main Frame
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left Frame (Camera Feed)
        left_frame = tk.Frame(main_frame, bg='white')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Camera Label
        self.camera_label = tk.Label(left_frame, text="Camera Feed", font=("Arial", 16))
        self.camera_label.pack()

        # Video Canvas
        self.video_canvas = tk.Canvas(left_frame, width=640, height=480, bg='lightgray')
        self.video_canvas.pack()

        # Camera Control Buttons
        camera_frame = tk.Frame(left_frame, bg='white')
        camera_frame.pack(pady=10)

        tk.Button(camera_frame, text="Start Camera", command=self.toggle_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(camera_frame, text="Capture", command=self.capture_image).pack(side=tk.LEFT, padx=5)

        # Right Frame (Detection Results)
        right_frame = tk.Frame(main_frame, bg='white')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Detection Results
        tk.Label(right_frame, text="Detection Results", font=("Arial", 16)).pack()

        self.result_text = tk.Text(right_frame, height=10, width=50)
        self.result_text.pack(pady=10)

        # Confidence and Debug Info
        self.debug_label = tk.Label(right_frame, text="Debug Information", font=("Arial", 12))
        self.debug_label.pack()

    def toggle_camera(self):
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.camera_active = True
        self.capture = cv2.VideoCapture(0)
        self.prediction_thread = threading.Thread(target=self.camera_thread)
        self.prediction_thread.start()

    def stop_camera(self):
        self.camera_active = False
        if self.capture:
            self.capture.release()
        if self.prediction_thread:
            self.prediction_thread.join()

    def camera_thread(self):
        while self.camera_active:
            ret, frame = self.capture.read()
            if ret:
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update canvas
                self.update_canvas(processed_frame)

                # Small delay to control frame rate
                time.sleep(0.03)

    def process_frame(self, frame):
        # Flip and resize frame
        frame = cv2.flip(frame, 1)
        
        # Draw detection region
        height, width = frame.shape[:2]
        box_size = min(height, width) // 2
        x = (width - box_size) // 2
        y = (height - box_size) // 2
        
        cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (0, 255, 0), 2)
        
        # Predict sign in ROI
        roi = frame[y:y+box_size, x:x+box_size]
        prediction, confidence = self.predict_sign(roi)
        
        if prediction:
            cv2.putText(frame, f"Sign: {prediction}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def predict_sign(self, image):
        try:
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, 0

            prediction = self.model.predict(np.expand_dims(processed_image, axis=0))
            predicted_class_idx = np.argmax(prediction)
            confidence = prediction[0][predicted_class_idx] * 100
            predicted_sign = self.class_indices[predicted_class_idx]

            return predicted_sign, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0

    def preprocess_image(self, image, target_size=(64, 64)):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_size)
            image = image.astype('float32') / 255.0
            return image
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def update_canvas(self, frame):
        # Convert frame to PhotoImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb_frame)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_canvas.image = imgtk

    def capture_image(self):
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                # Save image
                filename = filedialog.asksaveasfilename(defaultextension=".jpg")
                if filename:
                    cv2.imwrite(filename, frame)
                    messagebox.showinfo("Capture", f"Image saved to {filename}")

def main():
    root = tk.Tk()
    app = SignLanguageRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()