import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pytesseract
import cv2

# 🔧 Configuration Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🖼️ Configuration de la page
st.set_page_config(page_title="Lecture Numéro de Conteneur", layout="centered")
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #003366; border-radius: 10px;'>
    <h1 style='color: white;'>🚢 CNSTN - Détection de Conteneur</h1>
    <h4 style='color: white;'>Comparaison OCR : caractère par caractère vs bloc</h4>
</div>
""", unsafe_allow_html=True)

# 🚀 Chargement du modèle YOLOv8
@st.cache_resource
def load_model():
    return YOLO(r"C:\Pfe_Mokh++\Detection_Conteneur_Computer_Vision\PythonProject1\PythonProject1\runs\detect\train2\weights\best.pt")

model = load_model()

# 🔡 Méthode 1 : OCR par découpe verticale + binarisation
def detect_by_characters(image, bbox, num_chars=11):
    x1, y1, x2, y2 = map(int, bbox)
    box_width = x2 - x1
    left = x1
    right = x1 + int(box_width * 0.3)
    top, bottom = y1, y2
    cropped = image.crop((left, top, right, bottom))

    gray = cropped.convert("L").rotate(90, expand=True)
    open_cv = np.array(gray)
    binary = cv2.adaptiveThreshold(open_cv, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 10)
    rotated = Image.fromarray(binary)

    w, h = rotated.size
    step = h // num_chars
    final_text = ""

    for i in range(num_chars):
        char_img = rotated.crop((0, i * step, w, (i + 1) * step))
        char = pytesseract.image_to_string(
            char_img,
            config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip()
        if len(char) == 1 and char.isalnum():
            final_text += char

    return final_text

# 🔤 Méthode 2 : OCR global sur la zone détectée
def detect_by_block(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image.crop((x1, y1, x2, y2)).convert("L").rotate(90, expand=True)
    text = pytesseract.image_to_string(
        cropped,
        config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ).strip()
    return ''.join(filter(str.isalnum, text))

# 📥 Téléversement de l’image
uploaded_file = st.file_uploader("📁 Téléversez une image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Image originale", use_column_width=True)

    with st.spinner("🔍 Analyse en cours..."):
        results = model(np.array(image))
        annotated_img = results[0].plot()

        if results[0].boxes:
            for i, box in enumerate(results[0].boxes):
                bbox = box.xyxy[0]

                text_char = detect_by_characters(image, bbox, num_chars=11)
                text_block = detect_by_block(image, bbox)

                st.markdown(f"""
                ### 🧱 OCR caractère par caractère : <span style='color:green'><b>{text_char}</b></span>  
                ### 🚀 OCR bloc complet : <span style='color:blue'><b>{text_block}</b></span>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Aucun conteneur détecté.")

    st.image(annotated_img, caption="✅ Résultat YOLOv8", use_column_width=True)
