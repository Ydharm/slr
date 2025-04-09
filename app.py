import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import time
import pyttsx3
import threading

# Page configuration
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="✌️",
    layout="wide"
)

# Updated CSS with new styling
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/free-vector/white-abstract-background_23-2148810113.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    
    .stButton>button {
        width: 120px;
        height: 40px;
        margin: 3px;
        background-color: #2B6CB0;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
        font-size: 14px;
        padding: 0 5px;
    }

    .stColumns {
        gap: 0.5rem;
    }

    .stButton>button:hover {
        background-color: #1A4971;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .output-text {
        font-size: 24px;
        padding: 15px;
        border: 2px solid #2B6CB0;
        border-radius: 5px;
        min-height: 60px;
        background-color: rgba(255, 255, 255, 0.9);
        margin: 5px 0;
        color: #2B6CB0;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    .word-box {
        font-size: 20px;
        padding: 10px;
        border: 1px solid #2B6CB0;
        border-radius: 5px;
        margin: 5px;
        background-color: rgba(255, 255, 255, 0.9);
        color: #2B6CB0;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .title-text {
        color: #1A4971;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .subtitle-text {
        color: #2B6CB0;
        font-weight: bold;
        margin: 15px 0;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }

    .stMarkdown {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 10px;
        border-radius: 5px;
    }

    .stVideo {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background: #2B6CB0;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #1A4971;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_tts():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        return engine
    except Exception as e:
        st.error(f"TTS initialization error: {e}")
        return None

TTS_ENGINE = initialize_tts()

def speak_text(text):
    def _speak():
        global TTS_ENGINE
        if TTS_ENGINE and text:
            try:
                TTS_ENGINE.say(text)
                TTS_ENGINE.runAndWait()
            except Exception as e:
                st.error(f"Speech error: {e}")
    
    threading.Thread(target=_speak).start()

@st.cache_resource
def load_recognition_model():
    model = load_model('models/final_model.h5')
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return model, {v: k for k, v in class_indices.items()}

def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, target_size)
        image = image / 255.0
    return image

def predict_sign(image, model, class_indices):
    processed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class_idx = np.argmax(prediction)
    confidence = prediction[0][predicted_class_idx] * 100
    predicted_sign = class_indices[predicted_class_idx]
    return predicted_sign, confidence

def main():
    try:
        model, class_indices = load_recognition_model()
    except Exception as e:
        st.error("Error loading the model. Please check model files.")
        return

    st.markdown('<h1 class="title-text">Sign Language Recognition</h1>', unsafe_allow_html=True)

    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'saved_words' not in st.session_state:
        st.session_state.saved_words = []
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = ""
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = time.time()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h3 class="subtitle-text">Camera Feed</h3>', unsafe_allow_html=True)
        stframe = st.empty()
        camera_running = st.button("Start/Stop Camera")

    with col2:
        st.markdown('<h3 class="subtitle-text">Current Detection</h3>', unsafe_allow_html=True)
        text_display = st.empty()
        text_display.markdown(
            f'<div class="output-text">{st.session_state.current_text}</div>',
            unsafe_allow_html=True
        )

        col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns([1.2, 1, 1, 1, 0.8])
        
        with col_btn1:
            if st.button("Save Word"):
                if st.session_state.current_text:
                    st.session_state.saved_words.append(st.session_state.current_text)
                    st.session_state.current_text = ""

        with col_btn2:
            if st.button("Space"):
                st.session_state.current_text += " "

        with col_btn3:
            if st.button("Clear"):
                st.session_state.current_text = ""

        with col_btn4:
            if st.button("Speak"):
                speak_text(st.session_state.current_text)

        st.markdown('<h3 class="subtitle-text">Saved Words</h3>', unsafe_allow_html=True)
        words_container = st.container()
        for word in st.session_state.saved_words:
            words_container.markdown(
                f'<div class="word-box">{word}</div>',
                unsafe_allow_html=True
            )

    if camera_running:
        cap = cv2.VideoCapture(0)
        
        PREDICTION_COOLDOWN = 2.0
        SAME_PREDICTION_COOLDOWN = 3.0
        CONFIDENCE_THRESHOLD = 95

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            box_size = min(height, width) // 2
            x = (width - box_size) // 2
            y = (height - box_size) // 2
            
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (43, 108, 176), 2)

            current_time = time.time()
            time_since_last = current_time - st.session_state.last_prediction_time

            if time_since_last >= PREDICTION_COOLDOWN:
                roi = frame[y:y+box_size, x:x+box_size]
                if roi.size != 0:
                    predicted_sign, confidence = predict_sign(
                        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),
                        model,
                        class_indices
                    )

                    # Skip prediction if it contains "nothing" or random characters
                    if (confidence > CONFIDENCE_THRESHOLD and 
                        "nothing" not in predicted_sign.lower() and
                        predicted_sign.isalnum() and
                        (predicted_sign != st.session_state.last_prediction or 
                         time_since_last >= SAME_PREDICTION_COOLDOWN)):
                        
                        st.session_state.current_text += predicted_sign
                        st.session_state.last_prediction = predicted_sign
                        st.session_state.last_prediction_time = current_time
                        
                        text_display.markdown(
                            f'<div class="output-text">{st.session_state.current_text}</div>',
                            unsafe_allow_html=True
                        )

                        # Only show text on frame if it's a valid prediction
                        cv2.putText(frame, f"Sign: {predicted_sign}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (43, 108, 176), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (43, 108, 176), 2)
                        
                        remaining_time = max(0, PREDICTION_COOLDOWN - time_since_last)
                        cv2.putText(frame, f"Next detection in: {remaining_time:.1f}s", (10, 110),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (43, 108, 176), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if not camera_running:
                break

        cap.release()

if __name__ == "__main__":
    main()