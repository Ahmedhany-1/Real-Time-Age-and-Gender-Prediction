import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load models
gender_processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
gender_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
gender_model.eval()

age_processor = AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier")
age_model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
age_model.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def predict_age_gender(face_img):
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).resize((224, 224))

    # Gender
    gender_inputs = gender_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        gender_outputs = gender_model(**gender_inputs)
    gender_probs = torch.nn.functional.softmax(gender_outputs.logits, dim=1)
    gender_idx = gender_probs.argmax(-1).item()
    gender_label = gender_model.config.id2label[gender_idx]
    gender_conf = gender_probs[0][gender_idx].item()

    # Age
    age_inputs = age_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        age_outputs = age_model(**age_inputs)
    age_probs = torch.nn.functional.softmax(age_outputs.logits, dim=1)
    age_idx = age_probs.argmax(-1).item()
    age_label = age_model.config.id2label[age_idx]
    age_conf = age_probs[0][age_idx].item()

    return f"{gender_label} ({gender_conf*100:.1f}%), {age_label} ({age_conf*100:.1f}%)"
