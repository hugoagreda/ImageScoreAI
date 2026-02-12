from pathlib import Path
from PIL import Image
from .runtime_models import encode_image, score_embedding

VALID_EXT = {".webp", ".jpg", ".jpeg", ".png"}


# =====================================
# SCORE IMAGE (PATH VERSION)
# =====================================

def score_image(image_path: str, return_embedding: bool = False):

    img_path = Path(image_path).resolve()

    if not img_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    try:
        with Image.open(img_path) as img:
            image = img.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Error loading image {img_path}: {e}")

    embedding = encode_image(image)
    score_data = score_embedding(embedding)

    result = {
        "image_path": str(img_path),
        "score": float(score_data["score"]),
        "margin": float(score_data["margin"]),
    }

    if return_embedding:
        result["embedding"] = embedding

    return result


# =====================================
# SCORE IMAGE (PIL VERSION) ← API READY
# =====================================

def score_image_pil(image, return_embedding=False):

    embedding = encode_image(image)
    score_data = score_embedding(embedding)

    result = {
        "score": float(score_data["score"]),
        "margin": float(score_data["margin"]),
    }

    if return_embedding:
        result["embedding"] = embedding

    return result


# =====================================
# SCORE BATCH (DESKTOP / TESTING)
# =====================================

def score_folder(folder_path: str):

    folder = Path(folder_path).resolve()

    if not folder.exists():
        raise FileNotFoundError(folder)

    results = []

    for img_path in folder.iterdir():

        if img_path.suffix.lower() not in VALID_EXT:
            continue

        try:
            r = score_image(str(img_path))
            results.append(r)

        except Exception as e:
            print(f"⚠️ Error con {img_path}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)

    return results
