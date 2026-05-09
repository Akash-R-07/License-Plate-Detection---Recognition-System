# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re
import pytesseract
from ultralytics import YOLO
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

@st.cache_resource
def load_model():
    return YOLO("best.pt")

@st.cache_data
def load_database():
    data = [
        ["M906090K", "Rahul", "New York", "9876543210"],
        ["A12345BC", "Anjali Menon", "Los Angeles", "9123456780"],
        ["Z98X76YQ", "Arjun Kumar", "Chicago", "9988776655"],
        ["QW123ERT", "Sneha Pillai", "Houston", "9090909090"],
        ["LK890PLM", "Vivek Das", "Phoenix", "9012345678"],
        ["T56789GH", "Suresh Kumar", "San Francisco", "9123456781"],
        ["B45NM789", "Priya Shetty", "Seattle", "9988776652"],
        ["X1Y2Z3A4", "Amit Sharma", "Miami", "9090909093"]
    ]
    df = pd.DataFrame(data, columns=["Vehicle_Number", "Owner_Name", "Place", "Phone_Number"])
    return df

def process_image(image_path, model, df):
    img = cv2.imread(image_path)
    results = model(img)
    
    for r in results:
        boxes = r.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            plate = img[y1:y2, x1:x2]
            
            # Preprocessing
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # OCR
            text = pytesseract.image_to_string(thresh, config='--psm 7')
            text = re.sub(r'[^A-Z0-9]', '', text)
            plate_number = text.strip().upper()
            
            # Database lookup
            result = df[df["Vehicle_Number"] == plate_number]
            
            # Draw rectangle
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            
            return img, plate_number, result  
    
    return img, "", pd.DataFrame()   

# ------------------------- STREAMLIT UI -------------------------
st.title("License Plate Detection & Recognition System")
uploaded_file = st.file_uploader("Upload car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    model = load_model()
    df = load_database()
    
    annotated_img, plate_number, match = process_image("temp_image.jpg", model, df)
    
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    st.image(annotated_img_rgb, caption="Result", use_container_width=True)
    
    # Show result
    if plate_number:
        st.write(f"**Detected Number:** {plate_number}")
        if not match.empty:
            st.success("Vehicle Found")
            st.dataframe(match)
        else:
            st.error("No data found for this vehicle")
    else:
        st.warning("No license plate detected")