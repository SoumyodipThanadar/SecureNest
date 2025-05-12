import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Load label names
label_dict = {}
with open("trainer/labels.txt") as f:
    for line in f:
        key, val = line.strip().split(":")
        label_dict[int(key)] = val

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class SmartDoorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Door Lock")
        self.root.geometry("700x600")
        self.root.configure(bg="lightgray")

        self.label = tk.Label(root, text="AI Door Lock", font=("Helvetica", 24, "bold"), bg="lightgray")
        self.label.pack(pady=10)

        self.video_frame = tk.Label(root)
        self.video_frame.pack()

        self.status_label = tk.Label(root, text="Waiting for face...", font=("Arial", 16), fg="blue", bg="lightgray")
        self.status_label.pack(pady=10)

        self.identity_label = tk.Label(root, text="Detected: None", font=("Arial", 14), bg="lightgray")
        self.identity_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.lock_state = "Locked"

        self.run_video()

    def run_video(self):
        def video_loop():
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)

                identity = "Unknown"
                for (x, y, w, h) in faces:
                    id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    if conf < 70:
                        identity = label_dict.get(id_, "Unknown")
                        if self.lock_state != "Unlocked":
                            self.status_label.config(text="Door Unlocked ✅", fg="green")
                            self.identity_label.config(text=f"Detected: {identity}")
                            self.lock_state = "Unlocked"
                    else:
                        if self.lock_state != "Locked":
                            self.status_label.config(text="Door Locked ❌", fg="red")
                            self.identity_label.config(text="Detected: Unknown")
                            self.lock_state = "Locked"

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)

        threading.Thread(target=video_loop, daemon=True).start()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartDoorApp(root)
    root.mainloop()
