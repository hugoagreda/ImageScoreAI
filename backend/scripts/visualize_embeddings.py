import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import umap

# =====================
# CONFIG
# =====================

EMB_PATH = Path("../data/embeddings/realestate_embeddings.parquet")

# =====================
# LOAD
# =====================

print("\n📂 Cargando embeddings...")

df = pd.read_parquet(EMB_PATH)

print(f"Embeddings cargados: {len(df)}")

# convertir lista → numpy
X = np.vstack(df["embedding"].values)
y = df["final_quality"].values

print(f"Shape embeddings: {X.shape}")

# =====================
# UMAP
# =====================

print("\n🧠 Calculando proyección UMAP...")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

X_2d = reducer.fit_transform(X)

# =====================
# PLOT
# =====================

print("\n🎨 Dibujando scatter...")

colors = {
    "good": "green",
    "medium": "orange",
    "bad": "red"
}

plt.figure(figsize=(10, 8))

for label in np.unique(y):
    idx = y == label
    plt.scatter(
        X_2d[idx, 0],
        X_2d[idx, 1],
        label=label,
        alpha=0.7
    )

plt.title("Real Estate Visual Quality — Embedding Space")
plt.legend()
plt.grid(True)

plt.show()
