import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
from detector import detect_faces, predict_age_gender
from PIL import Image

st.set_page_config(page_title="Age & Gender Detection", layout="centered")

st.title("🧠 Age & Gender Detection")

option = st.radio("Choose input method:", ["Upload Image", "Webcam (Live)"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        faces = detect_faces(img)

        if not faces.any():
            st.warning("No faces detected.")
        else:
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                result = predict_age_gender(face)
                gender_label, gender_conf = result["gender"]
                age_label, age_conf = result["age"]

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label_text = f"{gender_label} ({gender_conf*100:.1f}%), {age_label} ({age_conf*100:.1f}%)"
                cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

elif option == "Webcam (Live)":
    st.info("Live detection using webcam. Press Stop to end.")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            faces = detect_faces(img)

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                result = predict_age_gender(face)
                gender_label, gender_conf = result["gender"]
                age_label, age_conf = result["age"]

                label = f"{gender_label}, {age_label}"
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )
