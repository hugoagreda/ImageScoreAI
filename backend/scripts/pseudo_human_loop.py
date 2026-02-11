import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import cv2

# =====================
# CONFIG
# =====================

EMB_PATH = Path("../data/embeddings/realestate_embeddings.parquet")
MODEL_PATH = Path("../models/quality_head.joblib")

SHOW_LIMIT = 50   # cuántas imágenes revisar

# =====================
# LOAD DATA
# =====================

print("\n📂 Cargando embeddings...")

df = pd.read_parquet(EMB_PATH)

data = joblib.load(MODEL_PATH)

model = data["model"]
le = data["label_encoder"]

X = np.vstack(df["embedding"].values)

# =====================
# PREDICT PROBA
# =====================

print("\n🧠 Calculando predicciones...")

proba = model.predict_proba(X)

df["predicted_label"] = le.inverse_transform(np.argmax(proba, axis=1))

# incertidumbre = 1 - confianza máxima
df["confidence"] = np.max(proba, axis=1)
df["uncertainty"] = 1 - df["confidence"]

# =====================
# RANKING INTELIGENTE
# =====================

df_sorted = df.sort_values(by="uncertainty", ascending=False)

print("\n🔥 Mostrando imágenes más inciertas primero")

# =====================
# LOOP HUMANO
# =====================

count = 0

for _, row in df_sorted.iterrows():

    if count >= SHOW_LIMIT:
        break

    img_path = Path(row["image_path"])

    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))

    if img is None:
        continue

    print("\n--------------------------")
    print(f"Predicción modelo: {row['predicted_label']}")
    print(f"Confianza: {row['confidence']:.2f}")
    print("Pulsa: 1=bad | 2=medium | 3=good | ESC=salir")

    cv2.imshow("Pseudo Human Loop", img)

    key = cv2.waitKey(0)

    if key == 27:
        break

    # aquí SOLO mostramos, aún no guardamos cambios
    count += 1

cv2.destroyAllWindows()

print("\n✅ Loop finalizado")
