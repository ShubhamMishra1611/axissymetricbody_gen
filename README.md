
# Steps to Generate Curve:

### 1. Profile Preparation:
- Use the `thing.ipynb` file to:
  1. Generate a new CSV file with `z` coordinates.
  2. Estimate values of `i` and `j` based on the mid-body explanation above.
- Alternatively, use Excel/Sheets to compute the `z` coordinates and find `i` and `j`. The idea of `i` and `j` is explained in the section of `Understanding ith and jth Argument in blockMeshDict File`
- Ensure these CSV files are ready:
  - `slopes_distances_1_4_1_8.csv`
  - `slopes_distances_14_16_14_12.csv`
- These files contain information essential for generating support curves, based on `bb2_hull` submarine mesh curves.
- These files can be generated as one wishes, and step to generate it is explained in: `Generating Slope-Distance Data for Curve`

### 2. Run the `blockMeshDict.py` File:
Command:
```bash
python BlockMeshDict_gen.py --length 70 --height 4 --i 267 --j 466
```
#### Command-Line Arguments:
| Flag            | Type   | Default       | Description                                          |
|------------------|--------|---------------|------------------------------------------------------|
| `--length`      | float  | `70.156`      | Length of the submarine.                            |
| `--height`      | float  | `4.799`       | Height (radius) of the submarine.                   |
| `--wedge_angle` | float  | `5`           | Wedge angle in degrees.                             |
| `--coord_file`  | string | `'sf75withz.csv'` | Path to the coordinate CSV file.                    |
| `--i`           | int    | `277`         | Index `i` for forebody split.                       |
| `--j`           | int    | `489`         | Index `j` for aftbody split.                        |
| `--output_file` | string | `'blockMeshDict'` | Output path of  BlockMeshDict file.                     |

---

## Understanding ith and jth Argument in `blockMeshDict` File

### Simply Put:
- **i**: Index in the `.xy` file where the mid-body begins, i.e., where the slope of `y` is `0`. `i` should be just less than this index and not greater.
- **j**: Index in the `.xy` file where the mid-body ends. `j` should be more than this index and not less.

### Why Can't `i` or `j` Be the Index of Any Point on the Mid-Body?
If `i` or `j` corresponds to a mid-body index:
- The code assumes all points before `i` are forebody and all points after `j` are aftbody.
- This misclassification includes some mid-body points in fore or aft body regions, resulting in issues since the slope in the mid-body is `0`.
- The mathematical models for generating support curves behave poorly with zero slopes, leading to wierd shapes.

---

## Z-Coordinate:
Though the `z` coordinate is saved in the CSV or XLSX file, it is recalculated during processing to maintain the required wedge angle.

---


## Generating Slope-Distance Data for Curve:
### Prerequisites:
- Have all files in `spline_profile_data` for each curve.
  Example: `1_4.csv` contains `x`, `y`, `z` for the `1_4` curve. This was done manually by me.

### Steps:
1. In `geometry_info_collector.py`:
   - Set correct file names for the curves you want to relate.
   - Understand and handle `[::-1]` inversion for aftbody curves as explained below.
2. Run:
   ```bash
   python geometry_info_collector.py
   ```

---

## Understanding Inversion (`[::-1]`) in `geometry_info_collector.py`:
The file generates a CSV with columns `m1`, `m2`, `d1`, `d2`:
- **m1**: Slopes of the first curve.
- **m2**: Slopes of the second curve.
- **d1**: Distance between points on the first curve.
- **d2**: Distance between points on the second curve.

For example:
For a file name slopes_distances_14_16_14_12.csv, 
- **m1** : slopes of 1_4 curve
- **m2**  = slopes of 1_8 curves
- **d1** =  distance between points of 1_4 curves
- **d2** = distance between points of 1_8 curves

Slope and distance contains enough information to reconstruct any curve, what matters is direction of reconstruction or one can simply associate it with initial point of curve. Following images gives a idea.

### Direction Matters:
When reconstructing curves, ensure the same direction:
- **Forebody**: No `[::-1]` needed.
- **Aftbody**: Use `[::-1]` for the actual aft curve file.

#### Example:
For the file `slopes_distances_14_16_14_12.csv`:
- File1 (`14_16.csv`): Aftbody curve (use `[::-1]`).
- File2 (`12_14.csv`): Reference curve.

Code:
```python
file1 = '14_16.csv'
file2 = '12_14.csv'
points_1 = read_and_drop_z(file1)[::-1]  # Inversion for aftbody.
points_2 = read_and_drop_z(file2)
```

---

## File Explanations:

### 1. `blockMeshDict.py`
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

### 2. `generate_curve.py`
Describe the process of generating and interpolating support curves.

### 3. `geometry_info_collector.py`
Explain how slope-distance relationships between curves are calculated and their relevance to reconstruction.

---