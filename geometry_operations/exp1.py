'''
EXP1:
There might be issue when we are generating curve after upsampling the slopes and distances
'''

import numpy as np
import matplotlib.pyplot as plt
from geometry_interpolators import method2_resample_data, method2_resample_data_with_distance_correction
from curve_generator import generate_curve_using_slope_and_distance

M = 20
N = 200

t = np.linspace(0, 2 * np.pi, M)

initial_points = np.vstack((t, np.cos(t))).T

def calculate_slope(points):
    deltas = np.diff(points, axis=0)
    slopes = deltas[:, 1] / deltas[:, 0]
    return slopes

def calculate_distances(points):
    deltas = np.diff(points, axis=0)
    distances = np.sqrt((deltas ** 2).sum(axis=1))
    return distances

m1 = calculate_slope(initial_points)
d1 = calculate_distances(initial_points)

def error_calculate(generated_curve, act_curve):
  error = np.linalg.norm(generated_curve - act_curve)
  return error


# resample_m1 = method2_resample_data_with_distance_correction(m1, N)
# resample_d1 = method2_resample_data_with_distance_correction(d1, N)

resample_m1, resample_d1 = method2_resample_data_with_distance_correction(m1, d1, N)


simple_curve = generate_curve_using_slope_and_distance(m1, d1)
resample_based_curve = generate_curve_using_slope_and_distance(resample_m1, resample_d1)
# print(f'{resample_based_curve.shape = }')

# Check total path length for original and resampled curves
# print(f"Original total distance: {np.sum(d1)}")
# print(f"Resampled total distance: {np.sum(resample_d1)}")

# plt.plot(d1, 'r--', label='actual slope')
# plt.plot(resample_d1, label = 'resampled slope')
plt.plot(initial_points[:, 0], initial_points[:, 1], 'r--', label='actual Curve (N points)')
plt.plot(simple_curve[:, 0], simple_curve[:, 1], 'b.', label='Curve simpply curve (N points)')
plt.plot(resample_based_curve[:, 0], resample_based_curve[:, 1], 'g-', label='Curve simpply curve (N points)')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Generation')
plt.show()
