import cv2
import numpy as np
import streamlit as st
from AI import detection_

st.set_page_config(
    page_title="NHẬN DIỆN BIỂN SỐ XE",
    layout="wide",
    initial_sidebar_state="expanded"
    )

st.title("NHẬN DIỆN BIỂN SỐ XE")
uploaded_file = st.file_uploader("Choose a image file")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    detection_(opencv_image)
    st.image(opencv_image, caption='Results', use_column_width=True)

