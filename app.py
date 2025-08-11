import streamlit as st
import cv2
import time

# Threshold for detection
thres = 0.5

# Load class names
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Model files
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Load the network
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Streamlit page config
st.set_page_config(page_title="Object Detection App", layout="wide")
st.title("📷 Real-Time Object Detection with OpenCV + Streamlit")
st.write("Model: SSD MobileNet v3 trained on COCO dataset")

# Start/Stop button
run = st.checkbox('Start Camera')

# Placeholder for the image
frame_placeholder = st.empty()

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 648)
cap.set(4, 448)
cap.set(10, 70)

while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to access webcam.")
        break

    # Detect objects
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(),
                        (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)),
                        (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Convert BGR to RGB for Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(img_rgb, channels="RGB")

    # To prevent high CPU usage
    time.sleep(0.02)

cap.release()
