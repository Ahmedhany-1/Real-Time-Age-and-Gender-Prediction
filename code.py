import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

# Load models
gender_processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
gender_model.eval()

age_processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
age_model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
age_model.eval()

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Prediction function with confidence
def predict_age_gender(face_img):
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).resize((224, 224))

    # Gender prediction
    gender_inputs = gender_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        gender_outputs = gender_model(**gender_inputs)
    gender_probs = torch.nn.functional.softmax(gender_outputs.logits, dim=1)
    gender_idx = gender_probs.argmax(-1).item()
    gender_label = gender_model.config.id2label[gender_idx]
    gender_conf = gender_probs[0][gender_idx].item()

    # Age prediction
    age_inputs = age_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        age_outputs = age_model(**age_inputs)
    age_probs = torch.nn.functional.softmax(age_outputs.logits, dim=1)
    age_idx = age_probs.argmax(-1).item()
    age_label = age_model.config.id2label[age_idx]
    age_conf = age_probs[0][age_idx].item()

    return f"{gender_label} ({gender_conf*100:.1f}%), {age_label} ({age_conf*100:.1f}%)"

# Webcam capture
def run_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            label = predict_age_gender(face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow("Webcam - Press Q to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Upload photo
def upload_photo():
    path = filedialog.askopenfilename()
    if not path:
        return

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        label = predict_age_gender(face_img)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Image Detection - Press Any Key to Close", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Tkinter GUI
root = tk.Tk()
root.title("Age & Gender Detection")

tk.Label(root, text="Select Mode").pack(pady=10)
tk.Button(root, text="📷 Use Webcam", width=25, command=run_webcam).pack(pady=10)
tk.Button(root, text="🖼 Upload Photo", width=25, command=upload_photo).pack(pady=10)

root.mainloop()
