import pandas as pd
import requests
from pathlib import Path
import time

# =========================
# CONFIGURACIÓN
# =========================

CSV_PATH = "../data/datasets/kaggle_prefiltered.csv"
OUTPUT_DIR = Path("../data/images/kaggle_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_DOWNLOADS = 100   # 👈 cuántas imágenes nuevas quieres descargar por ejecución
TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =========================
# CARGAR DATASET
# =========================

df = pd.read_csv(CSV_PATH, sep=",", engine="python")

# =========================
# DETECTAR ARCHIVOS YA EXISTENTES
# =========================

existing_files = {p.name for p in OUTPUT_DIR.glob("*.webp")}

downloaded = 0
skipped = 0
already_present = 0

print(f"📦 Imágenes ya existentes detectadas: {len(existing_files)}")

# =========================
# PIPELINE DE DESCARGA
# =========================

for idx, row in df.iterrows():

    if downloaded >= MAX_NEW_DOWNLOADS:
        break

    url = row["image"]
    filename = OUTPUT_DIR / f"img_{idx}.webp"

    # 🔒 Evitar redescargar imágenes ya existentes
    if filename.name in existing_files:
        already_present += 1
        continue

    try:
        r = requests.get(url, timeout=TIMEOUT, headers=HEADERS)

        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            with open(filename, "wb") as f:
                f.write(r.content)

            downloaded += 1
            existing_files.add(filename.name)

        else:
            skipped += 1

    except Exception:
        skipped += 1

    if downloaded % 100 == 0 and downloaded != 0:
        print(
            f"⬇️ nuevas descargadas: {downloaded} | "
            f"ya existentes: {already_present} | "
            f"saltadas: {skipped}"
        )

    # 🧠 educación básica al servidor (evita bloqueos)
    time.sleep(0.05)

# =========================
# RESULTADO FINAL
# =========================

print("\n✅ FIN DESCARGA INCREMENTAL")
print(f"Nuevas descargadas: {downloaded}")
print(f"Ya existentes: {already_present}")
print(f"Saltadas: {skipped}")