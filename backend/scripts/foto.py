import sys
from pathlib import Path
import requests
from PIL import Image
import cv2

# Permite ejecutar el archivo directamente
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.runtime_models import encode_image, score_embedding
from metrics.visual_metrics import compute_all_metrics


VALID_EXT = {".webp", ".jpg", ".jpeg", ".png"}

YOLO_MODEL = None


# =====================================
# PRINT HELPER
# =====================================

def print_scan_result(result):

    print("\n========== SCAN COMPLETO ==========")
    print(f"Imagen: {result['image_path']}")
    print(f"Score visual: {result['score']:.3f}")

    print("\n🧠 Métricas:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.2f}")

    print("\n🧩 Objetos detectados:")
    if result["detected_objects"]:
        print(", ".join(result["detected_objects"]))
    else:
        print("None")


# =====================================
# YOLO LOADER (LAZY)
# =====================================

def get_yolo():

    global YOLO_MODEL

    if YOLO_MODEL is None:
        from ultralytics import YOLO

        print("🚀 [RUNTIME] Loading YOLO model...")
        YOLO_MODEL = YOLO("backend/scripts/yolov8n.pt")
        print("✅ YOLO ready")

    return YOLO_MODEL


# =====================================
# DOWNLOAD IMAGE FROM URL
# =====================================

def download_temp_image(url: str) -> Path:

    temp_path = Path("temp_runtime_image.jpg")

    print("\n🌐 Descargando imagen desde URL...")

    r = requests.get(url, timeout=10)

    if r.status_code != 200:
        raise RuntimeError("No se pudo descargar la imagen")

    with open(temp_path, "wb") as f:
        f.write(r.content)

    return temp_path.resolve()


# =====================================
# SCORE IMAGE (CORE)
# =====================================

def score_image(image_path: str):

    img_path = Path(image_path).resolve()

    if not img_path.exists():
        raise FileNotFoundError(img_path)

    with Image.open(img_path) as img:
        image = img.convert("RGB")

    embedding = encode_image(image)
    score_data = score_embedding(embedding)

    return {
        "image_path": str(img_path),
        "score": float(score_data["score"]),
    }


# =====================================
# FULL SCAN
# =====================================

def scan_image(image_path: str):

    img_path = Path(image_path).resolve()

    pil_img = Image.open(img_path).convert("RGB")
    cv_img = cv2.imread(str(img_path))

    if cv_img is None:
        raise RuntimeError("cv2 no pudo leer la imagen")

    base = score_image(str(img_path))
    metrics = compute_all_metrics(cv_img)

    model = get_yolo()
    preds = model(str(img_path), verbose=False)[0]

    detected = []

    if preds.boxes is not None:
        detected = [model.names[int(c)] for c in preds.boxes.cls]

    result = {
        "image_path": base["image_path"],
        "score": base["score"],
        "metrics": metrics,
        "detected_objects": detected,
    }

    print_scan_result(result)

    return result


# =====================================
# INTERACTIVE TERMINAL MODE
# =====================================

def interactive_scan():

    print("\n🧠 Real Estate Visual Scanner")
    print("Introduce una URL o ruta local de imagen:\n")

    user_input = input("👉 Imagen: ").strip()

    if user_input.startswith("http"):
        img_path = download_temp_image(user_input)
    else:
        img_path = Path(user_input)

    scan_image(str(img_path))


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    interactive_scan()
