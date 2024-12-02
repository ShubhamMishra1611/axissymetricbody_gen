import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def calculate_slope(points):
    deltas = np.diff(points, axis=0)
    slopes = deltas[:, 1] / deltas[:, 0]  # slope = dy / dx
    return slopes

def calculate_distances(points):
    deltas = np.diff(points, axis=0)
    distances = np.sqrt((deltas ** 2).sum(axis=1))  # Euclidean distance formula
    return distances

def upsample_data(data, target_length):
    """
    Upsample data using linear interpolation to match the target length.
    """
    original_indices = np.arange(len(data))
    target_indices = np.linspace(0, len(data) - 1, target_length)
    
    # Perform linear interpolation
    interpolator = interp1d(original_indices, data, kind='linear', axis=0)
    return interpolator(target_indices)

def generate_new_curve(forebody_curve, m1_upsampled, m2_upsampled, d1_upsampled, d2_upsampled):
    # The first point of the new curve is the same as the first point of the forebody curve
    new_curve = [forebody_curve[0]]

    # Iterate over each consecutive pair of points in the forebody
    for i in range(1, len(forebody_curve)):
        # Forebody's slope and distance for the current segment
        m_forebody = (forebody_curve[i, 1] - forebody_curve[i-1, 1]) / (forebody_curve[i, 0] - forebody_curve[i-1, 0])
        d_forebody = np.sqrt((forebody_curve[i, 0] - forebody_curve[i-1, 0])**2 + (forebody_curve[i, 1] - forebody_curve[i-1, 1])**2)

        m1 = m1_upsampled[i-1]
        m2 = m2_upsampled[i-1]
        d1 = d1_upsampled[i-1]
        d2 = d2_upsampled[i-1]

        m_target = (m1 * m_forebody) / m2
        d_target = (d1 * d_forebody) / d2

        theta = np.arctan(m_target)
        delta_x = d_target * np.cos(theta)
        delta_y = d_target * np.sin(theta)

        new_x = new_curve[-1][0] + delta_x
        new_y = new_curve[-1][1] + delta_y

        new_curve.append([new_x, new_y])
    return np.array(new_curve)