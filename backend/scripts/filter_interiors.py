import cv2
import numpy as np
from pathlib import Path
import pandas as pd

# ======================================
# CONFIGURACIÓN
# ======================================

INPUT_DIR = Path("../data/images/kaggle_raw")
OUTPUT_CSV = Path("../data/datasets/interior_filter_pro.csv")

MAX_IMAGES = 251   # ajusta según necesites

# ======================================
# FEATURES BASE
# ======================================

def has_sky(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    mask = ((h > 90) & (h < 130))
    return np.sum(mask) / mask.size


def edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def line_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=40,
        maxLineGap=5,
    )

    if lines is None:
        return 0

    return len(lines) / (img.shape[0] * img.shape[1])


def texture_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()


def brightness_uniformity(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.var()


def center_texture(img):
    h, w, _ = img.shape
    crop = img[h//4:3*h//4, w//4:3*w//4]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray.std()


def color_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    p = hist / np.sum(hist)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def edge_direction_ratio(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    vx = np.mean(np.abs(sobelx))
    vy = np.mean(np.abs(sobely))

    if vx + vy == 0:
        return 0

    return vy / (vx + vy)


# ======================================
# NUEVAS FEATURES CROMÁTICAS (PRO)
# ======================================

def color_diversity(img, k=6):
    small = cv2.resize(img, (64, 64))
    data = small.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_RANDOM_CENTERS
    )

    unique_labels = len(np.unique(labels))
    return unique_labels / k


def saturation_variance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].std()


# ======================================
# SCORING CONTINUO PRO V3
# ======================================

def compute_indoor_score(f):

    score = 0

    score += (1 - min(f["sky"]*3, 1)) * 0.12
    score += min(f["edges"]*10, 1) * 0.12
    score += min(f["lines"]*90000, 1) * 0.12
    score += min(f["texture"]/80, 1) * 0.12
    score += min(f["center_tex"]/60, 1) * 0.12
    score += min(f["entropy"]/6, 1) * 0.10
    score += min(f["brightness"]/6000, 1) * 0.10

    # NUEVO: balance cromático
    score += (1 - f["color_div"]) * 0.10

    # NUEVO: estabilidad de saturación
    score += (1 - min(f["sat_var"]/80, 1)) * 0.10

    return round(score, 4)


# ======================================
# CARGAR CSV EXISTENTE
# ======================================

if OUTPUT_CSV.exists():
    df_results = pd.read_csv(OUTPUT_CSV)
else:
    df_results = pd.DataFrame()

processed = set(df_results["image_path"].values) if len(df_results) > 0 else set()

count = 0
rows = []

# ======================================
# PIPELINE PRINCIPAL
# ======================================

for img_path in INPUT_DIR.glob("*.webp"):

    if count >= MAX_IMAGES:
        break

    if str(img_path) in processed:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    f = {}

    f["sky"] = has_sky(img)
    f["edges"] = edge_density(img)
    f["lines"] = line_density(img)
    f["texture"] = texture_variance(img)
    f["brightness"] = brightness_uniformity(img)
    f["center_tex"] = center_texture(img)
    f["entropy"] = color_entropy(img)
    f["edir"] = edge_direction_ratio(img)
    f["color_div"] = color_diversity(img)
    f["sat_var"] = saturation_variance(img)

    indoor_score = compute_indoor_score(f)

    rows.append({
        "image_path": str(img_path),
        "indoor_score": indoor_score,
        "visual_class": "candidate",
        **f
    })

    count += 1

    if count % 50 == 0:
        print(f"Procesadas nuevas: {count}")

# ======================================
# GUARDAR RESULTADOS
# ======================================

if rows:
    df_new = pd.DataFrame(rows)
    df_results = pd.concat([df_results, df_new], ignore_index=True)
    df_results.to_csv(OUTPUT_CSV, index=False)

print("\n🔥 FILTER PRO V3 COMPLETADO")
print(f"Nuevas procesadas: {count}")
print(f"Total acumulado: {len(df_results)}")
