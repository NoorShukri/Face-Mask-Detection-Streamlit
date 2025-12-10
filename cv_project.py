import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import tempfile
import mediapipe as mp
from tensorflow.keras.layers import DepthwiseConv2D

# --- 1. Fix for MobileNetV2 Loading (From Notebook) ---
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Mask Guard",
    page_icon="🛡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Styling & Helpers ---
COLOR_MASK = (0, 255, 0)      # Pure Green
COLOR_NO_MASK = (255, 0, 0)   # Pure Red
COLOR_TEXT = (255, 255, 255)  # White Text

# --- Constants & Paths ---
MODEL_PATH = 'final_best_mask_model.h5' 
IMG_SIZE = (224, 224) 

# --- Load Model (Cached) ---
@st.cache_resource
def load_trained_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Detection & Prediction Function ---
def detect_and_predict(image, model, detection_confidence, model_selection_param):
    if isinstance(image, np.ndarray):
        image_np = image
    else:
        image_np = np.array(image.convert('RGB'))
    
    output_image = image_np.copy()
    h, w, c = output_image.shape
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_found = False
    
    with mp_face_detection.FaceDetection(min_detection_confidence=detection_confidence, model_selection=model_selection_param) as face_detection:
        results = face_detection.process(image_np)
        
        if results.detections:
            face_found = True
            for detection in results.detections:
                # Extract face bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                wd = int(bboxC.width * w)
                ht = int(bboxC.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                
                # Tight Crop 
                x_new = x
                y_new = y
                w_new = wd
                h_new = ht
                
                # Extract/Crop the face Region of Interest (ROI)
                face_roi = image_np[y_new:y_new+h_new, x_new:x_new+w_new]
                
                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    try:
                        # Preprocessing
                        processed_face = cv2.resize(face_roi, IMG_SIZE) # 224x224
                        processed_face = np.expand_dims(processed_face, axis=0)
                        processed_face = processed_face / 255.0 # Normalization
                        
                        # اPrediction
                        prediction = model.predict(processed_face, verbose=0)
                        # Unpack predictions (Mask vs. No Mask probabilities)
                        (mask, withoutMask) = prediction[0]
                        
                        # Determine class label and color
                        if mask > withoutMask:
                            label = "MASK"
                            color = COLOR_MASK
                            val = mask
                        else:
                            label = "NO MASK"
                            color = COLOR_NO_MASK
                            val = withoutMask
                        
                        label_text = f"{label} {val*100:.0f}%"
                        
                        # Draw bounding box on image
                        cv2.rectangle(output_image, (x, y), (x+wd, y+ht), color, 2)
                        
                        # Calculate text size for label background
                        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                        
                        # Adjust label position (ensure it fits within frame)
                        if y - 25 > 0:
                            text_y = y - 6
                            rect_start = (x, y - 25)
                            rect_end = (x + text_w + 10, y)
                        else:
                            text_y = y + text_h + 10
                            rect_start = (x, y)
                            rect_end = (x + text_w + 10, y + text_h + 20)

                        cv2.rectangle(output_image, rect_start, rect_end, color, -1)
                        cv2.putText(output_image, label_text, (x + 5, text_y), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_TEXT, 1)
                                    
                    except Exception as e:
                        pass
                        
    return output_image, face_found

# --- Main UI Layout ---
st.title("🛡 AI Mask Guard")
st.markdown("### Intelligent Face Mask Compliance System")

model = load_trained_model()

if model is None:
    st.warning("⚠ Model file missing. Please make sure you have 'final_best_mask_model.h5' in this directory.")
else:
    # --- Sidebar Design ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3028/3028630.png", width=80)
    st.sidebar.title("Control Panel")
    
    st.sidebar.markdown("---")
    app_mode = st.sidebar.selectbox("📂 Select Input Source", ["Upload Image", "Upload Video", "Live Webcam"])
    
    with st.sidebar.expander("⚙ Advanced Settings", expanded=False):
        range_option = st.radio(
            "Camera Range",
            ["Close (Laptop)", "Far (CCTV)"],
            index=0,
            help="Choose 'Far' for surveillance footage or distant crowds."
        )
        model_selection_param = 0 if "Close" in range_option else 1

        detection_confidence = st.slider(
            "AI Sensitivity", 
            min_value=0.1, max_value=0.9, value=0.3, 
            help="Lower this value to detect faces more aggressively."
        )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Model: MobileNetV2 (224x224)")

    # --- Mode 1: Image ---
    if app_mode == "Upload Image":
        st.info("📸 Upload a static image for analysis.")
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original")
                st.image(image, use_container_width=True)
            
            if st.button("🚀 Run Analysis", use_container_width=True):
                with st.spinner("Processing..."):
                    result_image, face_found = detect_and_predict(image, model, detection_confidence, model_selection_param)
                    with col2:
                        st.subheader("Result")
                        st.image(result_image, use_container_width=True)
                        
                        if not face_found:
                            st.error("⚠ No faces detected!")
                            st.caption("Try lowering the *AI Sensitivity* in the sidebar settings.")

    # --- Mode 2: Video ---
    elif app_mode == "Upload Video":
        st.info("🎬 Upload a video file for frame-by-frame analysis.")
        uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            
            st_frame = st.empty()
            
            col1, col2 = st.columns([1, 4])
            with col1:
                stop_video = st.button("Stop", type="primary")
            with col2:
                st.caption("Analyzing video stream...")

            while cap.isOpened() and not stop_video:
                ret, frame = cap.read()
                if not ret: break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_frame, _ = detect_and_predict(frame_rgb, model, detection_confidence, model_selection_param)
                st_frame.image(result_frame, channels="RGB", use_container_width=True)
            
            cap.release()

    # --- Mode 3: Webcam ---
    elif app_mode == "Live Webcam":
        st.info("🎥 Live surveillance mode active.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            run_camera = st.toggle('🔴 Start Camera', value=False)
        
        st_frame = st.empty()
        
        if run_camera:
            camera = cv2.VideoCapture(0) 
            
            if not camera.isOpened():
                st.error("🚨 Error: Could not access the webcam. Please check if another app is using it.")
            else:
                while run_camera:
                    ret, frame = camera.read()
                    if not ret:
                        st.error("Failed to read frame from camera.")
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result_frame, _ = detect_and_predict(frame_rgb, model, detection_confidence, model_selection_param)
                    st_frame.image(result_frame, channels="RGB", use_container_width=True)
                
                camera.release()
