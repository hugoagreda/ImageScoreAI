import os
import sys
from pathlib import Path

import cv2
import pandas as pd

# Ensure backend modules are importable when running as a script.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Import individual metric functions
from metrics.lighting import compute_lighting_score
from metrics.sharpness import compute_sharpness_score
from metrics.composition import compute_composition_score
from metrics.color import compute_color_score
from metrics.clutter import compute_clutter_score

# Import global aggregator
from scoring.aggregator import compute_global_score


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

# Root directory where processed images are stored
IMAGE_ROOT = "backend/data/images"

# Output CSV path
OUTPUT_CSV = "backend/data/datasets/visual_quality_dataset.csv"

# Human-defined ground truth scores per quality category
GROUND_TRUTH_SCORES = {
    "good": 85,
    "medium": 55,
    "bad": 25
}


# --------------------------------------------------
# 2. Dataset construction
# --------------------------------------------------

rows = []

for quality_label, target_score in GROUND_TRUTH_SCORES.items():
    folder_path = os.path.join(IMAGE_ROOT, quality_label)

    # Skip if folder does not exist
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # --------------------------------------------------
        # 3. Load image
        # --------------------------------------------------
        image = cv2.imread(image_path)

        # Skip unreadable or corrupted files
        if image is None:
            continue

        # --------------------------------------------------
        # 4. Compute individual metrics
        # --------------------------------------------------
        lighting_score = compute_lighting_score(image)
        sharpness_score = compute_sharpness_score(image)
        composition_score = compute_composition_score(image)
        color_score = compute_color_score(image)
        clutter_score = compute_clutter_score(image)

        # --------------------------------------------------
        # 5. Compute global score (baseline)
        # --------------------------------------------------
        global_score = compute_global_score(
            lighting=lighting_score,
            sharpness=sharpness_score,
            composition=composition_score,
            color=color_score,
            clutter=clutter_score
        )

        # --------------------------------------------------
        # 6. Store results
        # --------------------------------------------------
        rows.append({
            "image_name": filename,
            "lighting": lighting_score,
            "sharpness": sharpness_score,
            "composition": composition_score,
            "color": color_score,
            "clutter": clutter_score,
            "global_score": global_score,
            "target_score": target_score
        })


# --------------------------------------------------
# 7. Save dataset to CSV
# --------------------------------------------------

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Dataset successfully generated.")
print(f"Total samples: {len(df)}")
print(f"Saved to: {OUTPUT_CSV}")
