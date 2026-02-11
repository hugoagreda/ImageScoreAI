import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib
import random

# =====================
# CONFIG
# =====================

EMB_PATH = Path("../data/embeddings/realestate_embeddings.parquet")
MODEL_PATH = Path("../models/pairwise_ranker.joblib")

PAIR_SAMPLES = 5000   # número máximo de pares a generar

# ranking numérico
RANK_MAP = {
    "bad": 0,
    "medium": 1,
    "good": 2
}

# =====================
# LOAD DATA
# =====================

print("\n📂 Cargando embeddings...")

df = pd.read_parquet(EMB_PATH)

# eliminar filas sin label por seguridad
df = df[df["final_quality"].notna()].copy()

X = np.vstack(df["embedding"].values)
y = df["final_quality"].values

print(f"Embeddings cargados: {len(df)}")

# =====================
# GENERAR PARES
# =====================

print("\n🧠 Generando pares pairwise...")

pairs_X = []
pairs_y = []

indices = list(range(len(df)))

for _ in range(PAIR_SAMPLES):

    i, j = random.sample(indices, 2)

    rank_i = RANK_MAP[y[i]]
    rank_j = RANK_MAP[y[j]]

    # ignorar pares iguales
    if rank_i == rank_j:
        continue

    emb_i = X[i]
    emb_j = X[j]

    # diferencia DIRECTA (no invertimos dirección)
    diff = emb_i - emb_j

    # label = 1 si A mejor que B
    label = 1 if rank_i > rank_j else 0

    pairs_X.append(diff)
    pairs_y.append(label)

pairs_X = np.array(pairs_X)
pairs_y = np.array(pairs_y)

print(f"Pares generados válidos: {len(pairs_X)}")

# seguridad extra
if len(np.unique(pairs_y)) < 2:
    raise ValueError("❌ Solo se generó una clase en pairs_y. Revisa el dataset.")

# =====================
# TRAIN RANKER
# =====================

print("\n🚀 Entrenando pairwise ranker...")

model = LogisticRegression(
    max_iter=2000
)

model.fit(pairs_X, pairs_y)

# =====================
# SAVE MODEL
# =====================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\n✅ Pairwise ranker guardado en:")
print(MODEL_PATH)
