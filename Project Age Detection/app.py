import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="Age Detection", layout="wide")

st.title("👶 Age Detection using OpenCV DNN")
st.write("Detect face and predict age from image")

# Model paths - tere folder ke hisab se
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

# Age ranges jo model predict karta hai
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Check if model files exist
def check_files():
    files = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}")
        return False
    return True

@st.cache_resource
def load_models():
    """Load face and age detection models"""
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    return face_net, age_net

def detect_faces(net, frame, conf_threshold=0.7):
    """Detect faces in frame and return boxes"""
    frame_h, frame_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, face_boxes

def predict_age(face_img, age_net):
    """Predict age from face image"""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]
    return age

# Main App
if check_files():
    face_net, age_net = load_models()

    # Sidebar
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider("Face Confidence", 0.5, 1.0, 0.7, 0.05)

    # Image Selection
    option = st.radio("Choose Image Source:", ["Use kid1.jpg", "Upload New Image"])

    if option == "Use kid1.jpg":
        image_path = "kid1.jpg"
        if not os.path.exists(image_path):
            st.error("kid1.jpg not found")
            st.stop()
        image = Image.open(image_path)
    else:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is None:
            st.info("Please upload an image")
            st.stop()
        image = Image.open(uploaded_file)
        image_path = uploaded_file.name

    # Display original
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # Process button
    if st.button("Detect Age"):
        with st.spinner("Processing..."):
            # Convert PIL to OpenCV
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect faces
            frame_with_boxes, face_boxes = detect_faces(face_net, frame.copy(), conf_threshold)

            if len(face_boxes) == 0:
                st.warning("No face detected! Try lowering confidence threshold")
            else:
                # Predict age for each face
                result_frame = frame.copy()
                age_results = []

                for i, (x1, y1, x2, y2) in enumerate(face_boxes):
                    # Extract face with margin
                    face = result_frame[max(0, y1-20):min(y2+20, result_frame.shape[0]-1),
                                        max(0, x1-20):min(x2+20, result_frame.shape[1]-1)]

                    if face.size == 0:
                        continue

                    age = predict_age(face, age_net)
                    age_results.append(age)

                    # Draw box and label
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"Age: {age}"
                    cv2.putText(result_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                # Convert back to RGB for streamlit
                result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

                with col2:
                    st.subheader("Detection Result")
                    st.image(result_rgb, use_column_width=True)
                    st.success(f"Detected {len(face_boxes)} face(s)")
                    for idx, age in enumerate(age_results):
                        st.metric(f"Face {idx+1} Age", age)
else:
    st.stop()

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Models: Caffe Face Detector + Age Net")