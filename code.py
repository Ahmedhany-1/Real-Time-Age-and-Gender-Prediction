import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load gender model
gender_processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
gender_model.eval()

# Load age model
age_processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
age_model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
age_model.eval()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize((224, 224))

        # Gender prediction
        gender_inputs = gender_processor(images=face_pil, return_tensors="pt")
        with torch.no_grad():
            gender_outputs = gender_model(**gender_inputs)
        gender_label = gender_model.config.id2label[gender_outputs.logits.argmax(-1).item()]

        # Age prediction
        age_inputs = age_processor(images=face_pil, return_tensors="pt")
        with torch.no_grad():
            age_outputs = age_model(**age_inputs)
        age_label = age_model.config.id2label[age_outputs.logitss.argmax(-1).item()]

        label = f"{gender_label}, {age_label}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Webcam - Age & Gender Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
