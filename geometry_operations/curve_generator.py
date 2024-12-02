import numpy as np

def generate_curve_using_slope_and_distance(slope, distance, init_point = (0 ,1)):
  # check if slope and distance have same number of points
  if len(slope) != len(distance):
    raise ValueError("Slope and distance must have the same number of points.")
  
  curve = [init_point]

  for i in range(1, slope.shape[0]):
    theta = np.arctan(slope[i-1])
    delta_x = distance[i-1] * np.cos(theta)
    delta_y = distance[i-1] * np.sin(theta)

    new_point = (curve[-1][0] + delta_x, curve[-1][1] + delta_y)
    curve.append(new_point)
  
  return np.array(curve)


if __name__ == '__main__':
    M = 40
    N = 100

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
    
    import matplotlib.pyplot as plt
    slope = calculate_slope(initial_points)
    distance = calculate_distances(initial_points)

    curve = generate_curve_using_slope_and_distance(slope, distance)

    plt.plot(initial_points[:, 0], initial_points[:, 1], 'r.', label='actual Curve (N points)')
    plt.plot(curve[:, 0], curve[: , 1], 'b--', label='Original Curve (M points)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Curve Resampling')
    plt.show()