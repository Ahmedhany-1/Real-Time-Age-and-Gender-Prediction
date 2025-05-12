# 🧠Real-Time Age and Gender Detection Using Hugging Face and Streamlit

This example face is used as input to demonstrate the app’s analysis. The app will detect the face and draw a bounding box around it. It will then predict the person’s approximate age range and gender (male/female). The application combines live face detection with demographic prediction: it can run on a webcam stream or on uploaded images, drawing bounding boxes around detected faces and overlaying age/gender labels with confidence bars. Similar projects use pre-trained models for these tasks, for example noting that they “utilize pre-trained deep learning models to detect faces, estimate age, and classify gender”. Key features include:

* **Face Detection:** Uses OpenCV’s Haar cascade to locate faces in each frame. Detected faces are outlined with a box (e.g. “created a bounding box around it”) as a foundation for further analysis.
* **Gender Classification:** A Hugging Face Vision Transformer (`<span>rizvandwiki/gender-classification</span>`) predicts each face’s gender (male or female) in real time.
* **Age Classification:** A Hugging Face Vision Transformer (`<span>nateraw/vit-age-classifier</span>`) predicts the face’s age group (an approximate age range).
* **Webcam & Upload Modes:** The app supports streaming live video from the webcam (using `<span>streamlit-webrtc</span>`) or uploading static images for analysis. The user can switch between these modes in the interface.
* **Visual Output:** The app overlays bounding boxes and confidence bars on the video/image feed. Detected faces are annotated with the predicted age range and gender label along with a confidence score.

## 🚀 Features

- 📸 Real-time webcam support via **Streamlit WebRTC**
- 🖼️ Upload and analyze any image
- 👁️ Face detection with bounding boxes using OpenCV
- 🧑‍🎓 Age estimation using `nateraw/vit-age-classifier`
- 🚻 Gender classification using `rizvandwiki/gender-classification`
- 📊 Model confidence displayed as progress bars
- 🧩 Modular code with separate UI and core logic
- 🌐 Streamlit-based web interface

## 🧰 Technology Stack


| Layer          | Tool/Library                                                                                    |
| -------------- | ----------------------------------------------------------------------------------------------- |
| UI             | [Streamlit](https://streamlit.io/)                                                              |
| Webcam         | [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)                                 |
| Models         | [Hugging Face Transformers](https://huggingface.co)                                             |
| Age Model      | [`nateraw/vit-age-classifier`](https://huggingface.co/nateraw/vit-age-classifier)               |
| Gender Model   | [`rizvandwiki/gender-classification`](https://huggingface.co/rizvandwiki/gender-classification) |
| Face Detection | OpenCV Haar cascades                                                                            |

## 🛠️ Setup & Installation

1. **Clone the Repository:**
   ```
   git clone https://github.com/your-username/real-time-age-gender-detection.git
   cd real-time-age-gender-detection
   ```
2. **Create a Python Virtual Environment:**
   ```
   python3 -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies:** Install all required packages listed in `<span>requirements.txt</span>`:
   ```
   pip install -r requirements.txt
   ```

## 📁 Folder Structure

```peri
real-time-age-gender-detection/
├── app.py                      # Main Streamlit app (UI + logic)
├── detector.py                # Face detection and prediction logic
├── utils.py                   # Helper functions
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
├── models/                    # (Optional) Cached models
├── assets/                    # Images or icons for UI
├── pages/                     # Optional: Additional Streamlit pages
└── .streamlit/                # Streamlit config
```

## Usage Instructions📄

1. **Run the App:** Launch the Streamlit app with the command:
   ```
   streamlit run app.py
   ```
2. **Choose Input Mode:** In the web UI, select either **Webcam** mode to start the live camera stream (the browser will request camera access) or **Upload** mode to choose a static image file from your computer.
3. **View Results:** Once the mode is active, the app will display the input video stream or image. Faces in each frame will be detected, boxed, and annotated. You will see each detected face’s predicted age range and gender (male/female) with a confidence bar.
4. **Stop or Exit:** To end the webcam session, press the Stop button or close the window.

## 🌍 Deployment

You can deploy this app on:

* [Streamlit **Community **Cloud]()
* [Render](https://render.com)
* [Heroku](https://heroku.com) *(**requires **special **support **for **webcam)***

If deploying on Streamlit Cloud, be sure to:

* Add `streamlit-webrtc` to `requirements.txt`
* Upload a `.streamlit/secrets.toml` if using Hugging Face API tokens (for private models)

## 🤝 **Contributions**

**Contributions, **issues, **and **suggestions **are **welcome!
Feel **free **to **open **a **pull **request **or **contact **me.**
