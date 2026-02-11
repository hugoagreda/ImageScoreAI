import pandas as pd
import torch
from pathlib import Path
from PIL import Image
import open_clip

# =====================
# CONFIG
# =====================

CSV_PATH = Path("../data/datasets/interior_final_candidates.csv")
OUTPUT_PATH = Path("../data/embeddings/realestate_embeddings.parquet")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# =====================
# LOAD DATASET
# =====================

df = pd.read_csv(CSV_PATH)

df = df[df["final_quality"].notna()].copy()

print(f"\n📊 Imagenes a procesar: {len(df)}")

# =====================
# LOAD OPENCLIP
# =====================

print("\n🚀 Cargando modelo OpenCLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
)

model = model.to(DEVICE)
model.eval()

print(f"✅ Modelo cargado en {DEVICE}")

# =====================
# PIPELINE
# =====================

embeddings = []

for idx, row in df.iterrows():

    img_path = Path(row["image_path"]).resolve()

    if not img_path.exists():
        print(f"❌ No existe: {img_path}")
        continue

    try:

        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb.cpu().numpy()[0].tolist()   # 🔥 importante para parquet

        embeddings.append({
            "image_path": str(img_path),
            "final_quality": row["final_quality"],
            "embedding": emb
        })

    except Exception as e:
        print(f"⚠️ Error con {img_path}: {e}")

    if (idx + 1) % 50 == 0:
        print(f"📦 Procesadas: {idx+1}/{len(df)}")

# =====================
# SAVE
# =====================

print("\n💾 Guardando embeddings...")

df_emb = pd.DataFrame(embeddings)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_emb.to_parquet(OUTPUT_PATH, index=False)

print("\n✅ Embeddings guardados en:")
print(OUTPUT_PATH)

print(f"\n📊 Total embeddings creados: {len(df_emb)}")
