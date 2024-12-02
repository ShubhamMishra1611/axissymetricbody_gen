# generate_curve.py
### 1. **`calculate_slope(points)`**
   - **Purpose**: Computes the slope of line segments between consecutive points.
   - **Steps**:
     1. Compute differences (`deltas`) between consecutive points for x and y coordinates.
     2. Calculate the slope for each segment as the ratio of `dy/dx`.

---

### 2. **`calculate_distances(points)`**
   - **Purpose**: Calculates the Euclidean distance between consecutive points.
   - **Steps**:
     1. Compute differences (`deltas`) between consecutive points.
     2. Calculate the Euclidean distance for each segment using the formula: 
        \[
        \text{distance} = \sqrt{\Delta x^2 + \Delta y^2}
        \]

---

### 3. **`upsample_data(data, target_length)`**
   - **Purpose**: Increases the number of points in a dataset using linear interpolation to match a specified target length.
   - **Steps**:
     1. Generate indices for the original data.
     2. Create a set of evenly spaced target indices to match the desired target length.
     3. Use `interp1d` from `scipy` to interpolate the data along the new indices.

---

### 4. **`generate_new_curve(forebody_curve, m1_upsampled, m2_upsampled, d1_upsampled, d2_upsampled)`**
   - **Purpose**: Generates a new curve by transforming an input curve (`forebody_curve`) using upsampled slope and distance corrections.
   - **Steps**:
     1. Initialize the new curve with the first point of the `forebody_curve`.
     2. For each subsequent point:
        - Calculate the slope and distance of the current segment in the `forebody_curve`.
        - Use upsampled slope (`m1`, `m2`) and distance (`d1`, `d2`) corrections to compute a target slope and distance for the new segment.
        - Compute the angle (`theta`) from the target slope.
        - Calculate the x and y offsets (`delta_x`, `delta_y`) using the target distance and angle.
        - Append the new point (`new_x`, `new_y`) to the new curve.
     3. Return the resulting curve as a NumPy array.


# geometry_info_collector.py

### 1. **`plot_slopes(m1, m2, file_1, file_2)`**
   - **Purpose**: Plots the slopes `m1` and `m2` on the same graph for visual comparison.
   - **Key Details**:
     - `m1` and `m2` are the slope arrays corresponding to two curves from `file_1` and `file_2`.
---

### 2. **`read_and_drop_z(filename)`**
   - **Purpose**: Reads a CSV file containing point data and removes the z-coordinate to leave only x and y.
   - **Key Details**:
     - Returns the x and y columns as a NumPy array for further processing.

---

### 3. **`resample_curve(points, N)`**
   - **Purpose**: Adjusts a curve to have exactly `N` evenly spaced points while preserving its geometry.
   - **Key Details**:
     1. Compute distances between consecutive points.
     2. Calculate cumulative distances (curve length up to each point).
     3. Interpolate x and y coordinates along the cumulative curve length to create `N` evenly spaced points.
   - This ensures uniform sampling of the curve regardless of its original point density.

---

### 4. **`calculate_slope(points)`**
   - **Purpose**: Calculates the slopes (`dy/dx`) of line segments between consecutive points.

---

### 5. **`calculate_distances(points)`**
   - **Purpose**: Computes Euclidean distances between consecutive points.

---

### 6. **`process_files(points1, points2, output_file)`**
   - **Purpose**: Processes two sets of point data, calculates slopes and distances, and saves the results.
   - **Steps**:
     1. **Resample points**: 
        - The curve with fewer points is upsampled to match the number of points in the larger curve using `resample_curve`.
     2. **Calculate slopes and distances**:
        - For both curves, slopes (`m1`, `m2`) and distances (`d1`, `d2`) are calculated using `calculate_slope` and `calculate_distances`.
     3. **Save results**:
        - The slopes and distances are saved into a CSV file (`output_file`) for further analysis.

---

### **Main Execution**:
- **Files to Process**:
  - `14_16.csv`: Points for one curve (reversed with `[::-1]`).
  - `12_14.csv`: Points for another curve.
- **Output File**:
  - Results of slope and distance comparisons are saved to `slopes_distances_14_16_14_12.csv`.


# blockMeshDict_gen.py
This script generates a **blockMeshDict** file for a submarine geometry using input dimensions and coordinate data. Here's an overview of the steps it performs:

1. **Argument Parsing**: Reads input parameters (length, height, wedge angle, coordinate file path, indices for forebody and aftbody segmentation, and output file name) via command-line arguments.

2. **Coordinate Normalization**:
   - Reads and normalizes coordinate data from the provided CSV file.
   - Scales the coordinates to fit within the specified submarine dimensions.

3. **Forebody Curve Generation**:
   - Extracts forebody coordinates based on the specified index (`i`).
   - Generates BSpline curves and support curves using slope and distance data from a separate CSV file.
   - Scales and adjusts the curves for proper alignment.

4. **Aftbody Curve Generation**:
   - Extracts aftbody coordinates based on the specified index (`j`).
   - Similarly processes and generates BSpline curves for the aftbody using slope and distance data.
   - Rescales the curves for domain alignment.


# geometry_operations/geometry_interpolators.py

---

### **Functions**

---

#### 1. **`resample_curve(points, N)`**
- **Purpose**: Resamples a 2D curve to have exactly `N` points while maintaining its geometric structure.
- **Steps**:
  1. **Calculate Distances**: Computes the Euclidean distances between consecutive points.
  2. **Cumulative Arc Length**: Finds the cumulative length of the curve (arc length).
  3. **Target Lengths**: Generates `N` equally spaced positions along the total arc length.
  4. **Interpolation**: Interpolates x and y coordinates to match these new positions.
- **Output**: A 2D array of `N` points, where each row is `[x, y]`.

---

#### 2. **`method2_resample_data(data, target_length)`**
- **Purpose**: Performs linear interpolation to upsample a dataset to a desired length (`target_length`).
- **Steps**:
  1. Maps the original indices to a uniform array of `target_length`.
  2. Uses `scipy.interpolate.interp1d` to interpolate the data linearly across the new indices.
- **Output**: Resampled data with `target_length` elements.

---

#### 3. **`method2_resample_data_with_distance_correction(slope, distance, target_length)`**
- **Purpose**: Resamples slope and distance arrays to a target length and ensures the total distance remains consistent.
- **Steps**:
  1. Resamples `slope` and `distance` using `method2_resample_data`.
  2. Adjusts (`scales`) the distances to preserve the original total distance.
- **Output**: Resampled and corrected arrays for `slope` and `distance`.

---

#### 4. **`scale_curve_y_to_range(curve, target_min, target_max)`**
- **Purpose**: Scales the y-coordinates of a curve to fit within a specified range `[target_min, target_max]`.
- **Steps**:
  1. Extracts y-coordinates and determines their current min/max values.
  2. Normalizes the y-coordinates to the range [0, 1].
  3. Scales them to the target range `[target_min, target_max]`.
  4. Updates the curve with the scaled y-coordinates.
- **Output**: Curve with y-coordinates adjusted to the specified range.

---

#### 5. **`scale_curve_x_to_range(curve, target_min, target_max)`**
- **Purpose**: Similar to `scale_curve_y_to_range`, but scales the x-coordinates of a curve.
- **Output**: Curve with x-coordinates adjusted to the specified range.
