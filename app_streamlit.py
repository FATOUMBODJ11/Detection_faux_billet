import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ----------------------
# CONFIGURATION GÉNÉRALE
# ----------------------
st.set_page_config(
    page_title="Détection de faux billets",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de ton API FastAPI
API_URL = "http://127.0.0.1:8000/predict_csv"

# ----------------------
# EN-TÊTE
# ----------------------
st.title("💵 Détection de faux billets")
st.markdown(
    """
    Cette application permet de :
    - 📂 Uploader un fichier **CSV**
    - 🚀 Envoyer les données à l'API **FastAPI**
    - 📊 Visualiser les **prédictions et statistiques**
    """
)

st.markdown("---")

# ----------------------
# UPLOAD DE FICHIER
# ----------------------
uploaded_file = st.file_uploader("📂 Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    with st.expander("👀 Aperçu des données importées", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    # Bouton de lancement
    if st.button("🚀 Lancer les prédictions", type="primary"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            results = response.json()

            # Transformer la réponse en DataFrame
            predictions = pd.DataFrame({
                "prediction": results["predictions"],
                "probability": results["probabilities"]
            })

            # ----------------------
            # AFFICHAGE DES RÉSULTATS
            # ----------------------
            st.success("✅ Prédictions effectuées avec succès !")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 Table des prédictions")
                st.dataframe(predictions, use_container_width=True)

            with col2:
                st.subheader("📊 Statistiques")
                stats = predictions["prediction"].value_counts()
                st.write(stats)

                fig, ax = plt.subplots(figsize=(5,3))
                stats.plot(kind="bar", color=["green", "red"], ax=ax)
                ax.set_ylabel("Nombre")
                ax.set_title("Répartition des prédictions")
                st.pyplot(fig)

        else:
            st.error("❌ Erreur lors de l'appel à l'API. Vérifie qu'elle est en cours d'exécution.")

else:
    st.info("👉 Merci d'importer un fichier CSV pour commencer.")
