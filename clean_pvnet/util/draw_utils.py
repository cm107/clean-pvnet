import cv2
import numpy as np

def draw_corners(img: np.ndarray, corner_2d: np.ndarray, color: tuple=(255, 0, 0), thickness: int=2) -> np.ndarray:
    result = img.copy()
    indexes = [
        (0, 1), (1, 3), (3, 2), (2, 0), (0, 4), (4, 6), (6, 2),
        (5, 4), (4, 6), (6, 7), (7, 5), (5, 1), (1, 3), (3, 7)
    ]
    corner_data = corner_2d.tolist()
    for start_idx, end_idx in indexes:
        result = cv2.line(
            img=result,
            pt1=tuple([int(val) for val in corner_data[start_idx]]), pt2=tuple([int(val) for val in corner_data[end_idx]]),
            color=color, thickness=thickness
        )
    return result

def draw_pts2d(img: np.ndarray, pts2d: np.ndarray, color: tuple=(255, 0, 0), radius: int=2) -> np.ndarray:
    result = img.copy()
    for x, y in pts2d.tolist():
        result = cv2.circle(
            img=result, center=(int(x), int(y)),
            radius=radius, color=color, thickness=-1
        )
    return result