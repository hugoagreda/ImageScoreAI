import pandas as pd
import shutil
from pathlib import Path

# ======================================
# CONFIG
# ======================================

CSV_PATH = Path("../data/datasets/interior_filter_pro.csv")

PREVIEW_DIR = Path("../data/images/filter_preview")
INTERIORS_DIR = PREVIEW_DIR / "interiors"
REJECTED_DIR = PREVIEW_DIR / "rejected"

INTERIORS_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)

MAX_COPY = 100        # 👈 número máximo de imágenes a copiar
THRESHOLD = 0.70      # 👈 umbral indoor_score

# ======================================
# CARGAR CSV
# ======================================

df = pd.read_csv(CSV_PATH)

total_rows = len(df)

copied_total = 0
copied_interiors = 0
copied_rejected = 0
missing_files = 0

print("\n📊 INICIANDO PREVIEW DEL FILTRO")
print(f"Total registros en CSV: {total_rows}")
print(f"Umbral indoor_score: {THRESHOLD}")
print("-" * 40)

# ======================================
# PIPELINE
# ======================================

for _, row in df.iterrows():

    if copied_total >= MAX_COPY:
        break

    src = Path(row["image_path"])

    if not src.exists():
        missing_files += 1
        continue

    indoor_score = row["indoor_score"]

    if indoor_score < THRESHOLD:
        dst = INTERIORS_DIR / src.name
        copied_interiors += 1
    else:
        dst = REJECTED_DIR / src.name
        copied_rejected += 1

    if not dst.exists():
        shutil.copy2(src, dst)
        copied_total += 1

    if copied_total % 50 == 0:
        print(
            f"📦 Copiadas: {copied_total} | "
            f"interiors: {copied_interiors} | "
            f"rejected: {copied_rejected}"
        )

# ======================================
# RESUMEN FINAL
# ======================================

if copied_total > 0:
    pct_interiors = (copied_interiors / copied_total) * 100
    pct_rejected = (copied_rejected / copied_total) * 100
else:
    pct_interiors = 0
    pct_rejected = 0

print("\n✅ PREVIEW COMPLETADO")
print("-" * 40)
print(f"Total copiadas: {copied_total}")
print(f"Interiores: {copied_interiors} ({pct_interiors:.2f}%)")
print(f"Rechazadas: {copied_rejected} ({pct_rejected:.2f}%)")
print(f"Archivos no encontrados: {missing_files}")
print("-" * 40)
