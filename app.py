import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detector import detect_faces, predict_age_gender

st.set_page_config(page_title="Age & Gender Detection", layout="centered")

st.title("🧠 Age & Gender Detection")

option = st.radio("Choose input method:", ["Upload Image", "Webcam (Experimental)"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        faces = detect_faces(img)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            label = predict_age_gender(face)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected", use_column_width=True)

elif option == "Webcam (Experimental)":
    st.warning("Live webcam preview is not natively supported in Streamlit. Use 'Upload Image' for better experience.")
