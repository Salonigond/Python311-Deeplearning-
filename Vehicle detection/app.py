import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import os

st.set_page_config(page_title="Vehicle Detection", layout="wide")

st.title("🚗 Vehicle Detection using Haar Cascade")
st.write("Processing: car-video.mp4")

# Check if files exist
VIDEO_PATH = "car-video.mp4"
CASCADE_PATH = "cars.xml"

if not os.path.exists(VIDEO_PATH):
    st.error(f"{VIDEO_PATH} not found in folder")
    st.stop()

if not os.path.exists(CASCADE_PATH):
    st.error(f"{CASCADE_PATH} not found in folder. Download from OpenCV repo")
    st.stop()

# Load Haar Cascade
@st.cache_resource
def load_cascade():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        st.error("cars.xml is corrupted or invalid")
        return None
    return cascade

car_cascade = load_cascade()

# Sidebar Settings
st.sidebar.header("Detection Settings")
scale_factor = st.sidebar.slider("Scale Factor", 1.01, 1.5, 1.1, 0.01)
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 3, 1)
min_size = st.sidebar.slider("Min Size", 20, 100, 30, 5)

# Main Display Areas
col1, col2 = st.columns(2)
with col1:
    st.subheader("Detection Output")
    frame_placeholder = st.empty()
with col2:
    st.subheader("Statistics")
    stats_placeholder = st.empty()

# Control Buttons
start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

if start_button and car_cascade is not None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        st.stop()
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info(f"Video: {total_frames} frames @ {fps} FPS")
    
    car_count_total = 0
    frame_count = 0
    start_time = time.time()
    running = True
    
    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.success("Video processing completed!")
            break
        
        if stop_button:
            running = False
            break
        
        frame_count += 1
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect cars
        cars = car_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=(min_size, min_size)
        )
        
        # Draw bounding boxes
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Car {len(cars)}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        car_count_total += len(cars)
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Update statistics
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        progress = frame_count / total_frames if total_frames > 0 else 0
        
        with stats_placeholder.container():
            st.metric("Frame", f"{frame_count}/{total_frames}")
            st.metric("Cars in Current Frame", len(cars))
            st.metric("Total Cars Detected", car_count_total)
            st.metric("Processing Speed", f"{current_fps:.1f} FPS")
            st.progress(progress)
        
        time.sleep(1/fps) # Match video speed
    
    cap.release()
    st.success("Processing stopped")

else:
    st.info("Click 'Start Detection' button to begin processing car-video.mp4")
   