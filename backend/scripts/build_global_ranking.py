import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# =====================
# CONFIG
# =====================

EMB_PATH = Path("../data/embeddings/realestate_embeddings.parquet")
MODEL_PATH = Path("../models/pairwise_ranker.joblib")
OUTPUT_PATH = Path("../data/embeddings/realestate_global_ranking.parquet")

# mostrar paths completos en consola
pd.set_option('display.max_colwidth', None)

# =====================
# LOAD DATA
# =====================

print("\n📂 Cargando embeddings...")

df = pd.read_parquet(EMB_PATH)
model = joblib.load(MODEL_PATH)

X = np.vstack(df["embedding"].values)

print(f"Embeddings: {X.shape}")

# =====================
# GLOBAL RANKING SCORE
# =====================

print("\n🧠 Calculando ranking global...")

scores = []

# Cada imagen compite contra todas
for i in range(len(X)):

    emb_i = X[i]
    wins = 0

    for j in range(len(X)):

        if i == j:
            continue

        diff = (emb_i - X[j]).reshape(1, -1)

        pred = model.predict(diff)[0]

        wins += pred

    scores.append(wins)

df["global_rank_score"] = scores

# =====================
# SORT
# =====================

df_sorted = df.sort_values(by="global_rank_score", ascending=False)

print("\n🔥 Top 10 ranking global (paths completos):\n")

for _, row in df_sorted.head(10).iterrows():
    print(f"{row['global_rank_score']}  |  {row['image_path']}")

# =====================
# SAVE
# =====================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_sorted.to_parquet(OUTPUT_PATH, index=False)

print("\n✅ Ranking global guardado en:")
print(OUTPUT_PATH)
