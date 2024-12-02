import numpy as np
from scipy.interpolate import interp1d

def resample_curve(points, N):
    """
    Resample a 2D curve to have N points while retaining the curve's geometry.
    
    Parameters:
    points (numpy.ndarray): A 2D numpy array of shape (M, 2), where each row is [x, y].
    N (int): The desired number of points after resampling.

    Returns:
    numpy.ndarray: A 2D numpy array of shape (N, 2) with the resampled points.
    """
    # Step 1: Calculate the distances between consecutive points
    deltas = np.diff(points, axis=0)
    assert deltas.shape == (points.shape[0]-1, points.shape[1])
    distances = np.sqrt((deltas ** 2).sum(axis=1))  # Euclidean distances between points
    
    # Step 2: Calculate the cumulative arc length
    cumulative_lengths = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative_lengths[-1]  # Total length of the curve
    
    # Step 3: Create an array of N equally spaced arc lengths (target positions)
    target_lengths = np.linspace(0, total_length, N)
    
    # Step 4: Interpolate to find new points at these arc lengths
    new_points = np.empty((N, 2))
    new_points[:, 0] = np.interp(target_lengths, cumulative_lengths, points[:, 0])  # x-coordinates
    new_points[:, 1] = np.interp(target_lengths, cumulative_lengths, points[:, 1])  # y-coordinates
    
    return new_points


def method2_resample_data(data, target_length):
    """
    Upsample data using linear interpolation to match the target length.
    """
    original_indices = np.arange(len(data))
    target_indices = np.linspace(0, len(data) - 1, target_length)
    
    # Perform linear interpolation
    interpolator = interp1d(original_indices, data, kind='linear', axis=0)
    return interpolator(target_indices)

def method2_resample_data_with_distance_correction(slope, distance, target_length):
    # Store original total distance
    original_total_distance = np.sum(distance)
    
    # Resample both slope and distance
    resampled_slope = method2_resample_data(slope, target_length)
    resampled_distance = method2_resample_data(distance, target_length)
    
    # Scale distances to maintain total path length
    resampled_total_distance = np.sum(resampled_distance)
    scale_factor = original_total_distance / resampled_total_distance
    resampled_distance *= scale_factor
    
    return resampled_slope, resampled_distance

def scale_curve_y_to_range(curve, target_min, target_max):
    """
    Scale the y-coordinates of a curve to fit within a target range [target_min, target_max]
    
    Parameters:
    curve: np.array of shape (N, 2) containing x,y coordinates
    target_min: float, minimum desired y value
    target_max: float, maximum desired y value
    
    Returns:
    np.array of shape (N, 2) with scaled y coordinates
    """
    # Extract y coordinates
    y = curve[:, 1]
    
    # Get current min and max
    current_min = np.min(y)
    current_max = np.max(y)
    
    # Avoid division by zero in case all y values are the same
    if current_max == current_min:
        scaled_y = np.full_like(y, (target_max + target_min) / 2)
    else:
        # Scale to [0,1] first
        scaled_y = (y - current_min) / (current_max - current_min)
        
        # Then scale to target range
        scaled_y = scaled_y * (target_max - target_min) + target_min
    
    # Create new curve with scaled y values
    scaled_curve = np.copy(curve)
    scaled_curve[:, 1] = scaled_y
    
    return scaled_curve
def scale_curve_x_to_range(curve, target_min, target_max):
    """
    Scale the y-coordinates of a curve to fit within a target range [target_min, target_max]
    
    Parameters:
    curve: np.array of shape (N, 2) containing x,y coordinates
    target_min: float, minimum desired y value
    target_max: float, maximum desired y value
    
    Returns:
    np.array of shape (N, 2) with scaled y coordinates
    """
    # Extract y coordinates
    y = curve[:, 0]
    
    # Get current min and max
    current_min = np.min(y)
    current_max = np.max(y)
    
    # Avoid division by zero in case all y values are the same
    if current_max == current_min:
        scaled_y = np.full_like(y, (target_max + target_min) / 2)
    else:
        # Scale to [0,1] first
        scaled_y = (y - current_min) / (current_max - current_min)
        
        # Then scale to target range
        scaled_y = scaled_y * (target_max - target_min) + target_min
    
    # Create new curve with scaled y values
    scaled_curve = np.copy(curve)
    scaled_curve[:, 0] = scaled_y
    
    return scaled_curve



if __name__ == '__main__':
    M = 10  # Original number of points
    N = 200   # Desired number of points

    t = np.linspace(0, 2 * np.pi, M)
    curve = np.vstack((t, np.cos(t))).T  # Original points

    resampled_curve = resample_curve(curve, N)
    resampled_curve2 = method2_resample_data(curve, N)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(curve[:, 0], curve[:, 1], 'bo-', label='Original Curve (M points)')
    plt.plot(resampled_curve[:, 0], resampled_curve[:, 1], 'r--', label='Resampled Curve (N points)')
    plt.plot(resampled_curve2[:, 0], resampled_curve2[:, 1], 'g.', label='Resampled Curve (N points)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Curve Resampling')
    plt.show()
