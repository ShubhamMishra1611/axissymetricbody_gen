import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_slopes(m1, m2, file_1, file_2):
    """
    Plot the slopes m1 and m2 on the same plot for comparison.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(m1, label=f'Slope m1 ({file_1})', color='blue', marker='o')
    # plt.plot(m2, label=f'Slope m2 ({file_2})', color='red', marker='x')
    plt.xlabel('Index (Consecutive Points)')
    plt.ylabel('Slope (dy/dx)')
    plt.title('Comparison of Slopes (m1 vs m2)')
    plt.legend()
    plt.grid(True)
    plt.show()

def read_and_drop_z(filename):
    """
    Reads the CSV file and drops the z-coordinate.
    
    Returns:
    numpy.ndarray: A 2D array with only x and y columns.
    """
    data = pd.read_csv(filename, header=None, delim_whitespace=True)
    return data[[0, 1]].to_numpy()  # Return only x and y columns

def resample_curve(points, N):
    """
    Resample a 2D curve to have N points while retaining the curve's geometry.
    
    Returns:
    numpy.ndarray: A 2D numpy array of shape (N, 2) with the resampled points.
    """
    deltas = np.diff(points, axis=0)
    distances = np.sqrt((deltas ** 2).sum(axis=1))  # Euclidean distances between points
    cumulative_lengths = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative_lengths[-1]  # Total length of the curve
    target_lengths = np.linspace(0, total_length, N)
    new_points = np.empty((N, 2))
    new_points[:, 0] = np.interp(target_lengths, cumulative_lengths, points[:, 0])
    new_points[:, 1] = np.interp(target_lengths, cumulative_lengths, points[:, 1])
    return new_points

def calculate_slope(points):
    """
    Calculate slope (dy/dx) between consecutive points.
    
    Returns:
    numpy.ndarray: An array of slopes between consecutive points.
    """
    deltas = np.diff(points, axis=0)
    slopes = deltas[:, 1] / deltas[:, 0]  # slope = dy / dx
    return slopes

def calculate_distances(points):
    """
    Calculate the Euclidean distances between consecutive points.

    Returns:
    numpy.ndarray: An array of Euclidean distances between consecutive points.
    """
    deltas = np.diff(points, axis=0)
    distances = np.sqrt((deltas ** 2).sum(axis=1))  # Euclidean distance formula
    return distances

def process_files(points1, points2, output_file):
    """
    Process two CSV files and calculate slope and distance, then save the results.
    
    output_file (str): Path to the output CSV file where results will be saved.
    """
    
    
    # Step 2: Identify which file has more points and resample the one with fewer points
    M = max(len(points1), len(points2))
    N = min(len(points1), len(points2))
    
    if len(points1) == M:
        points2_resampled = resample_curve(points2, M)
        points1_resampled = points1
    else:
        points1_resampled = resample_curve(points1, M)
        points2_resampled = points2
    
    # plt.plot(points1_resampled[:, 0], points1_resampled[:, 1], "o-", label = "support curve")
    # plt.plot(points2_resampled[:, 0], points2_resampled[:, 1], "o-", label = "actual curve")
    # for i, (x, y) in enumerate(points1_resampled):
    #     plt.text(x, y, str(i), fontsize=9, ha='right', va='bottom')

    # for i, (x, y) in enumerate(points2_resampled):
    #     plt.text(x, y, str(i), fontsize=9, ha='right', va='bottom')
    # plt.xlabel('Index (Consecutive Points)')
    # plt.ylabel('points')
    # plt.title('comparision of points')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # exit(0)
    # Step 3: Calculate slopes and distances for both sets of points
    # x1 = points1_resampled[:, 0]
    # x2 = points2_resampled[:, 0]

    m1 = calculate_slope(points1_resampled)
    m2 = calculate_slope(points2_resampled)
    
    d1 = calculate_distances(points1_resampled)
    d2 = calculate_distances(points2_resampled)

    # Step 4: Save the results to a new CSV file
    result_df = pd.DataFrame({
        'm1': m1,
        'm2': m2,
        'd1': d1,
        'd2': d2
    })

    result_df.to_csv(output_file, index=False)

# run here
file1 = '14_16.csv'
file2 = '12_14.csv'
points_1 = read_and_drop_z(file1)[::-1] # [::-1] is there to invert the profile. Please refer to readme.md to understand when to use it.
points_2 = read_and_drop_z(file2)
output_file = 'slopes_distances_14_16_14_12.csv'

process_files(points_1, points_2, output_file)
