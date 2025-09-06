from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib


#  Charger le modèle et le scaler

model_scaler = joblib.load("lr_billet_scaler_29_08_2025.sav")
model = model_scaler['model']
scaler = model_scaler['scaler']

# Colonnes utilisées par le modèle
expected_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']


#  Créer l'API FastAPI

app = FastAPI(title="Détection de Faux Billets")

# Activer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines pour dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





#  Endpoint pour CSV

@app.post("/predict_csv/")
async def predict_billet_csv(file: UploadFile = File(...)):
    try:
        # Lire CSV
        df = pd.read_csv(file.file)

        # Vérifier et réordonner colonnes
        if not all(col in df.columns for col in expected_columns):
            return {"error": f"Colonnes attendues : {expected_columns}, colonnes reçues : {list(df.columns)}"}

        df = df[expected_columns]

        # Remplacer valeurs manquantes par la moyenne
        df = df.fillna(df.mean())

        # Standardiser
        X_scaled = scaler.transform(df)

        # Prédictions
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

        return {
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist()
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
#  Endpoint test
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de détection de faux billets !"}