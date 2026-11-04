import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------
# Configuration de la page (doit être la première commande Streamlit)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Détection de la tuberculose",
    page_icon="🫁",
    layout="centered",
)

# ---------------------------------------------------------------
# Paramètres issus de l'entraînement
# ---------------------------------------------------------------
MODEL_PATH = "vgg16_model_finetuned.h5"  # fichier du modèle, placé dans le dépôt GitHub à côté de app.py

IMG_SIZE = (150, 150)
CLASSES = ["Normal", "Tuberculosis"]


# ---------------------------------------------------------------
# Chargement du modèle
# @st.cache_resource : le modèle est chargé une seule fois,
# puis réutilisé à chaque interaction (sinon il serait rechargé à chaque clic)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Fichier du modèle introuvable : {MODEL_PATH}. Vérifiez qu'il est bien dans le dépôt GitHub.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ---------------------------------------------------------------
# Prétraitement + prédiction (identique à la version Gradio)
# ---------------------------------------------------------------
def predict(model, image):
    image = image.convert("RGB")                                # gris/RGBA -> 3 canaux
    image = image.resize(IMG_SIZE, Image.NEAREST)               # même interpolation que Keras
    image_array = np.asarray(image, dtype=np.float32) / 255.0   # même rescale
    image_array = np.expand_dims(image_array, axis=0)           # (1, 150, 150, 3)

    p_tb = float(model.predict(image_array, verbose=0)[0][0])
    return {CLASSES[1]: p_tb, CLASSES[0]: 1.0 - p_tb}


# ---------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------
st.title("🫁 Détection de la tuberculose sur radiographie thoracique")
st.write(
    "Téléversez une radiographie thoracique pour estimer la probabilité de tuberculose."
)
st.warning("Outil pédagogique : il ne remplace pas un diagnostic médical.")

model = load_model()

uploaded_file = st.file_uploader(
    "Radiographie thoracique",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_image, col_result = st.columns(2)

    # Colonne de gauche : l'image téléversée
    with col_image:
        st.image(image, caption="Image téléversée")

    # Colonne de droite : le résultat
    with col_result:
        with st.spinner("Analyse en cours..."):
            probas = predict(model, image)

        # Classe prédite = classe avec la plus grande probabilité
        prediction = max(probas, key=probas.get)

        st.subheader("Résultat")
        if prediction == "Tuberculosis":
            st.error(f"**{prediction}** ({probas[prediction]:.1%})")
        else:
            st.success(f"**{prediction}** ({probas[prediction]:.1%})")

        # Probabilités des deux classes, de la plus élevée à la plus faible
        for classe, proba in sorted(probas.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{classe} : {proba:.1%}")
            st.progress(proba)
else:
    st.info("Aucune image téléversée pour le moment.")

st.caption("DIT & AfriAI Solutions · Modèle VGG16 (transfer learning)")
