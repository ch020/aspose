import numpy as np
from typing import Dict, Tuple, TypeVar

A = TypeVar('A')

def vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculates the vector difference (b - a) between two points."""
    return b - a

def yaw(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates the angle (in degrees) between projections of v1 and v2 onto the X-Z plane"""
    v1xz, v2xz = v1[[0, 2]], v2[[0, 2]]
    v1xz /= max(np.linalg.norm(v1xz), 1e-8)
    v2xz /= max(np.linalg.norm(v2xz), 1e-8)
    return np.degrees(np.arccos(np.clip(np.dot(v1xz, v2xz), -1.0, 1.0)))

def bucket(value: float, *bins: Tuple[float, float, A]) -> A:
    """Return the label for the interval containing value."""
    if np.isnan(value):
        return np.nan
    value = round(value, 1)
    for lo, hi, label in bins:
        if lo <= value <= hi:
            return label
    raise ValueError(f"{value} outside bins")

# BASMI-10 Look-up Tables
_B_TRAGUS  = [(-np.inf,  9.99, 0),(10,12.9,1),(13,15.9,2),(16,18.9,3),
              (19,21.9,4),(22,24.9,5),(25,27.9,6),(28,30.9,7),
              (31,33.9,8),(34,36.9,9),(37,  np.inf,10)]

_B_LSF     = [(20, np.inf,0),(18,19.9,1),(15.9,17.9,2),(13.8,15.8,3),
              (11.7,13.7,4),(9.6,11.6,5),(7.5,9.5,6),(5.4,7.4,7),
              (3.3,5.3,8),(1.2,3.2,9),(-np.inf,1.19,10)]

_B_IMD     = [(120, np.inf,0),(110,119.9,1),(100,109.9,2),(90,99.9,3),
              (80,89.9,4),(70,79.9,5),(60,69.9,6),(50,59.9,7),
              (40,49.9,8),(30,39.9,9),(-np.inf,29.9,10)]

_B_CR      = [(85, np.inf,0),(76.6,84.9,1),(68.1,76.5,2),(59.6,68,3),
              (51.1,59.5,4),(42.6,51,5),(34.1,42.5,6),(25.6,34,7),
              (17.1,25.5,8),(8.6,17,9),(-np.inf,8.5,10)]
