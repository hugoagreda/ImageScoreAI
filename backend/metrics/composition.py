import cv2
import numpy as np


def _score_angle_deviation(deviations: np.ndarray, max_deviation: float) -> float:
    if deviations.size == 0:
        return 50.0

    mean_deviation = float(np.mean(deviations))
    score = 100 - (mean_deviation / max_deviation) * 100
    return float(np.clip(score, 0, 100))


def compute_composition_score(image: np.ndarray) -> float:
    """
    Computes a composition and geometric alignment score for an image.

    The score evaluates:
    - Vertical line alignment
    - Horizontal leveling (camera tilt)
    - Framing balance

    Returns:
        float: composition score normalized between 0 and 100
    """

    # --------------------------------------------------
    # 1. Convert image to grayscale and detect edges
    # --------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # --------------------------------------------------
    # 2. Detect line segments
    # --------------------------------------------------
    height, width = edges.shape
    min_line_length = max(20, int(min(height, width) * 0.1))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=10
    )

    vertical_deviation = []
    horizontal_deviation = []

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue

            angle = np.degrees(np.arctan2(dy, dx))
            angle = angle % 180

            if 75 <= angle <= 105:
                vertical_deviation.append(abs(90 - angle))
            elif angle <= 15 or angle >= 165:
                horizontal_deviation.append(min(angle, 180 - angle))

    vertical_deviation = np.array(vertical_deviation, dtype=float)
    horizontal_deviation = np.array(horizontal_deviation, dtype=float)

    # --------------------------------------------------
    # 3. Score vertical and horizontal alignment
    # --------------------------------------------------
    vertical_score = _score_angle_deviation(vertical_deviation, max_deviation=15.0)
    horizontal_score = _score_angle_deviation(horizontal_deviation, max_deviation=10.0)

    # --------------------------------------------------
    # 4. Score framing balance using edge distribution
    # --------------------------------------------------
    edge_points = np.column_stack(np.where(edges > 0))
    if edge_points.size == 0:
        balance_score = 50.0
    else:
        ys, xs = edge_points[:, 0], edge_points[:, 1]
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))

        norm_dx = abs(center_x - (width / 2)) / (width / 2)
        norm_dy = abs(center_y - (height / 2)) / (height / 2)
        imbalance = (norm_dx + norm_dy) / 2

        balance_score = 100 - imbalance * 100
        balance_score = float(np.clip(balance_score, 0, 100))

    # --------------------------------------------------
    # 5. Combine components into final score
    # --------------------------------------------------
    composition_score = (
        0.4 * vertical_score +
        0.3 * horizontal_score +
        0.3 * balance_score
    )

    return float(np.clip(composition_score, 0, 100))
