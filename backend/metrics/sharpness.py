import cv2
import numpy as np


def compute_sharpness_score(image: np.ndarray) -> float:
    """
    Computes a sharpness score for an image.

    The score is based on the variance of the Laplacian,
    which measures the amount of high-frequency detail
    (edges and texture).

    Returns:
        float: sharpness score normalized between 0 and 100
    """

    # --------------------------------------------------
    # 1. Convert image to grayscale
    # --------------------------------------------------
    # Sharpness is related to intensity changes, not color.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # 2. Apply Laplacian operator
    # --------------------------------------------------
    # The Laplacian highlights regions with rapid intensity changes (edges).
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # --------------------------------------------------
    # 3. Compute variance of the Laplacian
    # --------------------------------------------------
    # High variance → many edges → sharp image
    # Low variance → few edges → blurred image
    laplacian_variance = laplacian.var()

    # --------------------------------------------------
    # 4. Normalize variance into a 0–100 score
    # --------------------------------------------------
    # These thresholds are empirical but realistic for typical images.
    # They can be adjusted later if needed.
    min_var = 50      # very blurry
    max_var = 1000    # very sharp

    sharpness_score = (
        (laplacian_variance - min_var) /
        (max_var - min_var)
    ) * 100

    # Clip score to valid range
    sharpness_score = float(np.clip(sharpness_score, 0, 100))

    return sharpness_score
