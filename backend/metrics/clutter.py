import cv2
import numpy as np


def compute_clutter_score(image: np.ndarray) -> float:
    """
    Computes a visual clutter score for an image.

    The score evaluates:
    - Edge density as a proxy for visual complexity
    - Amount of fine-grained visual information

    Higher scores indicate cleaner, more visually simple scenes.

    Returns:
        float: clutter score normalized between 0 and 100
    """

    # --------------------------------------------------
    # 1. Convert image to grayscale
    # --------------------------------------------------
    # Clutter is related to structure, not color.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # 2. Detect edges
    # --------------------------------------------------
    # Edges represent visual transitions and object boundaries.
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # --------------------------------------------------
    # 3. Compute edge density
    # --------------------------------------------------
    # Edge density = proportion of pixels classified as edges.
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    edge_density = edge_pixels / total_pixels

    # --------------------------------------------------
    # 4. Normalize edge density into a clutter score
    # --------------------------------------------------
    # High edge density → cluttered scene → lower score
    # Low edge density → clean scene → higher score
    max_edge_density = 0.15  # empirical upper bound for interiors

    clutter_score = 100 - (edge_density / max_edge_density) * 100
    clutter_score = float(np.clip(clutter_score, 0, 100))

    return clutter_score
