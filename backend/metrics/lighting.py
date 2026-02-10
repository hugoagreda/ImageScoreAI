import cv2
import numpy as np


def compute_lighting_score(image: np.ndarray) -> float:
    """
    Computes a lighting quality score for an image.

    The score evaluates:
    - Global brightness level
    - Light distribution uniformity
    - Presence of underexposed or overexposed regions

    Returns:
        float: lighting score normalized between 0 and 100
    """

    # --------------------------------------------------
    # 1. Convert image to grayscale
    # --------------------------------------------------
    # Lighting analysis is based on intensity, not color.
    # Grayscale conversion simplifies the problem.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # 2. Compute basic brightness statistics
    # --------------------------------------------------
    # Mean intensity represents overall exposure level.
    mean_intensity = np.mean(gray)

    # Standard deviation reflects how evenly light is distributed.
    std_intensity = np.std(gray)

    # --------------------------------------------------
    # 3. Detect extreme intensity regions
    # --------------------------------------------------
    # Very dark pixels (underexposed)
    dark_pixels_ratio = np.sum(gray < 30) / gray.size

    # Very bright pixels (overexposed)
    bright_pixels_ratio = np.sum(gray > 220) / gray.size

    # --------------------------------------------------
    # 4. Score individual components
    # --------------------------------------------------

    # Ideal mean brightness is assumed to be around mid-gray (≈127)
    brightness_score = 100 - abs(mean_intensity - 127) * 100 / 127
    brightness_score = np.clip(brightness_score, 0, 100)

    # Lower standard deviation means more uniform lighting
    uniformity_score = 100 - std_intensity
    uniformity_score = np.clip(uniformity_score, 0, 100)

    # Penalize images with large extreme regions
    extreme_penalty = (dark_pixels_ratio + bright_pixels_ratio) * 100

    # --------------------------------------------------
    # 5. Combine components into final lighting score
    # --------------------------------------------------
    lighting_score = (
        0.5 * brightness_score +
        0.3 * uniformity_score -
        0.2 * extreme_penalty
    )

    # Ensure score is within expected range
    lighting_score = float(np.clip(lighting_score, 0, 100))

    return lighting_score
