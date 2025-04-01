import numpy as np

def calculate_vector(p1, p2):
    return np.array([p2['x'] - p1['x'], p2['y'] - p1['y'], p2['z'] - p1['z']])

def calculate_yaw_angle(v1, v2):
    xz_1 = np.array([v1[0], v1[2]])
    xz_2 = np.array([v2[0], v2[2]])
    unit_1 = xz_1 / np.linalg.norm(xz_1)
    unit_2 = xz_2 / np.linalg.norm(xz_2)

    dot_product = np.clip(np.dot(unit_1, unit_2), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg