import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from runtime.runtime_score import score_image_pil

app = FastAPI()


@app.post("/score")
async def score(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = score_image_pil(image)

    return result
