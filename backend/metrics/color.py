import cv2
import numpy as np


def compute_color_score(image: np.ndarray) -> float:
    """
    Computes a color balance and saturation score for an image.

    The score evaluates:
    - Presence of strong color casts (white balance issues)
    - Under-saturation or over-saturation

    Returns:
        float: color quality score normalized between 0 and 100
    """

    # --------------------------------------------------
    # 1. Convert image to LAB color space (for color balance)
    # --------------------------------------------------
    # LAB separates luminance (L) from chromatic components (A, B)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # --------------------------------------------------
    # 2. Analyze color balance
    # --------------------------------------------------
    # In a well-balanced image, the mean of A and B channels
    # should be close to the neutral value (128).
    a_mean = np.mean(a_channel)
    b_mean = np.mean(b_channel)

    # Compute deviation from neutral gray
    color_cast_deviation = abs(a_mean - 128) + abs(b_mean - 128)

    # Normalize deviation into score
    max_cast_deviation = 40.0  # empirical tolerance
    balance_score = 100 - (color_cast_deviation / max_cast_deviation) * 100
    balance_score = np.clip(balance_score, 0, 100)

    # --------------------------------------------------
    # 3. Convert image to HSV color space (for saturation)
    # --------------------------------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)

    # --------------------------------------------------
    # 4. Analyze saturation level
    # --------------------------------------------------
    mean_saturation = np.mean(saturation)

    # Ideal saturation range (empirical but realistic)
    ideal_saturation = 80

    saturation_score = 100 - abs(mean_saturation - ideal_saturation) * 100 / ideal_saturation
    saturation_score = np.clip(saturation_score, 0, 100)

    # --------------------------------------------------
    # 5. Combine balance and saturation scores
    # --------------------------------------------------
    color_score = 0.6 * balance_score + 0.4 * saturation_score
    color_score = float(np.clip(color_score, 0, 100))

    return color_score
