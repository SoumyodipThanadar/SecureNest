import cv2
import numpy as np
from PIL import Image
import os

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = []
    ids = []
    label_dict = {}
    label_count = 0

    dataset_path = 'dataset'

    for folder in os.listdir(dataset_path):
        label_dict[label_count] = folder
        folder_path = os.path.join(dataset_path, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, 'uint8')
            faces.append(img_np)
            ids.append(label_count)

        label_count += 1

    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer/trainer.yml')

    with open("trainer/labels.txt", "w") as f:
        for k, v in label_dict.items():
            f.write(f"{k}:{v}\n")

    print("[INFO] Training complete and model saved.")

if __name__ == "__main__":
    train_recognizer()
