from ultralytics import YOLO
import pandas as pd
from pathlib import Path

# =====================
# CONFIG
# =====================

CSV_PATH = Path("../data/datasets/interior_filter_pro.csv")
OUTPUT_CSV = Path("../data/datasets/interior_semantic.csv")

MODEL = YOLO("yolov8n.pt")  # modelo ligero

# clases COCO que nos interesan
INTERIOR_CLASSES = [
    "bed",
    "couch",
    "chair",
    "dining table",
    "tv",
    "potted plant",
]

# =====================
# LOAD DATA
# =====================

VISUAL_THRESHOLD = 0.70

df = pd.read_csv(CSV_PATH)

results = []

for _, row in df.iterrows():

    img_path = row["image_path"]

    preds = MODEL(img_path, verbose=False)[0]

    detected_names = [MODEL.names[int(c)] for c in preds.boxes.cls] if preds.boxes else []

    has_interior_object = any(obj in INTERIOR_CLASSES for obj in detected_names)

    results.append({
        "image_path": img_path,
        "indoor_score": row["indoor_score"],
        "has_semantic_interior": int(has_interior_object),
        "detected_objects": ",".join(detected_names)
    })

df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)

print("✅ YOLO semantic filter done")
