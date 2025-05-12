import cv2
import os

def capture_images(name):
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    folder_path = f"dataset/{name}"
    os.makedirs(folder_path, exist_ok=True)

    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{folder_path}/{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Capturing Faces', img)

        if cv2.waitKey(1) == 27 or count >= 30:  # ESC or 30 images
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {count} images saved to {folder_path}")

if __name__ == "__main__":
    username = input("Enter your name: ")
    capture_images(username)
