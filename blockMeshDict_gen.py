import numpy as np
import pandas as pd
import math
import argparse
import sys

import matplotlib.pyplot as plt
from generate_curve import *
from geometry_info_collector import plot_slopes
from geometry_operations.curve_generator import generate_curve_using_slope_and_distance
from geometry_operations.geometry_interpolators import resample_curve, method2_resample_data_with_distance_correction, scale_curve_x_to_range, scale_curve_y_to_range

PLOT_SHOW = True
PRINT_LOGS = True

def cutoff_non_ascending(y, data):
    # Find the index where y stops being ascending
    for i in range(1, len(y)):
        if y[i] < y[i - 1]:
            return data[:i]  # Cut off from that point onward
    return data

def make_y_ascending(y):
    y_fixed = y.copy()
    for i in range(1, len(y_fixed)):
        if y_fixed[i] < y_fixed[i - 1]:  # If not ascending
            y_fixed[i] = y_fixed[i - 1] + 1e-5  # Increment slightly to maintain order
    return y_fixed
def make_x2_ascending(y):
    y_fixed = y.copy()
    for i in range(1, len(y_fixed)):
        if y_fixed[i] < y_fixed[i - 1]:  # If not ascending
            y_fixed[i] = y_fixed[i - 1] + 1e-5  # Increment slightly to maintain order
    return y_fixed

def convert_curve(x):
    # dy_dx = x[1:] - x[:-1]
    return x**(2/3)

def make_y_descending(y):
    y_fixed = y.copy()
    for i in range(1, len(y_fixed)):
        if y_fixed[i] > y_fixed[i - 1]:  # If not descending
            y_fixed[i] = y_fixed[i - 1] - 1e-5  # Decrement slightly to maintain order
    return y_fixed

def create_block_mesh_dict(sub_length, sub_height, wedge_angle, coord_file, i, j, output_file):
    '''
    Generates blockMeshDict file.
    Args:
    sublength: int = Length of the submarine you want in the blockMeshDict
    sub_hight: int = Hight of submarine you want in the blockMeshDict
    wedge_angle: int = Wedge angle
    coord_file: path of the coord_file, which should be a xlsx format
    i: int = This index defines upto which forebody coordinate is there. You can get this value from simple visual inspection. 
                Try to get value of index before mid_body starts to ensure generated mesh is proper.
    j: int = This index defines upto which aftbody starts. You can get this value from simple visual inspection. 
                Try to get value of index after mid_body ends to ensure generated mesh is proper.

    i and j are most important numbers for perfect mesh generation.
    
    '''
    print(coord_file)
    # Step 1: Read and normalize the coordinates from the CSV file
    # coordinates = pd.read_excel(coord_file) # Uncomment this line if file original file is excel
    coordinates = pd.read_csv(coord_file)
    x_min, x_max = coordinates['X'].min(), coordinates['X'].max()
    y_min, y_max = coordinates['Y'].min(), coordinates['Y'].max()
    z_min, z_max = coordinates['z'].min(), coordinates['z'].max()
    if PRINT_LOGS:
        print(f'{y_min = }')
        print(f'{y_max = }')
        print(f'{x_min = }')
        print(f'{x_max = }')
        print(f"solution: ({(y_max - y_min) * 10/(x_max - x_min)})")
    y_scale = (y_max - y_min) * 10/(x_max - x_min)

    # Check if the 'X' column is sorted
    is_sorted = coordinates['X'].equals(coordinates['X'].sort_values())

    if is_sorted:
        print("The 'X' coordinates are sorted.")
    else:
        print("The 'X' coordinates are not sorted.")



    if PLOT_SHOW:
        plt.plot(coordinates['X'], coordinates['Y'])
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Coordinates plot")
        plt.show()

    # Min-max normalization
    coordinates['X'] = (coordinates['X'] - x_min) / (x_max - x_min) * sub_length # here we are defining the length of submarine after normalization
    coordinates['Y'] = (coordinates['Y'] - y_min) / (y_max - y_min) * y_scale
    coordinates['z'] = (coordinates['z'] - z_min) / (z_max - z_min) * y_scale

    coordinates = coordinates.astype(np.float128)

    
    # since normalization is happening 
    # Uncomment if want to change the sub_height and sub_length
    # sub_height = 2
    # sub_length = 10
    
    # Step 4: Calculate domain values
    domain_height = np.float128(5 * sub_height)  # m2 = 10, b2 = 0
    domain_left_gap = np.float128(2.5 * sub_length)  # m1 = 2.5, b1 = 0
    domain_top_z = np.float128(np.tan(np.radians(wedge_angle) / 2) * domain_height)

    # Step 2: Forebody (BSpline 1 8 and BSpline 1 6)
    print("GENERATION::: FOREBODY")
    forebody_coords = coordinates.iloc[:i].copy() # Here we extract all coordinates before ith index, which is basically forebody
    forebody_coords['z'] = forebody_coords['Y'] * np.tan(np.radians(wedge_angle) / 2) # getting z coordinates
    bspline_1_6 = forebody_coords.copy()
    bspline_1_8 = forebody_coords[['X', 'Y']].to_numpy()
    plt.plot(bspline_1_8[:, 0], bspline_1_8[:, 1], 'b.')
    plt.show()
    # exit(0)

    # From here code for generating support curve of forebody starts
    slope_distance_df_forebody = pd.read_csv('slopes_distances_1_4_1_8.csv') # File where slope distance for forebody is located
    support_curve_slope = slope_distance_df_forebody['m1'].to_numpy() # getting required columns from the slopes_distances_1_4_1_8.csv
    actual_curve_slope = slope_distance_df_forebody['m2'].to_numpy()
    support_curve_gaps = slope_distance_df_forebody['d1'].to_numpy()
    actual_curve_gaps = slope_distance_df_forebody['d2'].to_numpy()

    forebody_coords = forebody_coords.to_numpy()
    forebody_coords = forebody_coords[:, :2]

    # This function returns upsamples or increses the number of points in the 
    support_curve_m1_upsampled, support_curve_d1_upsampled = method2_resample_data_with_distance_correction(
        support_curve_slope,
        support_curve_gaps,
        len(bspline_1_8)
    )
    actual_curve_slope_upsampled, actual_curve_gaps_upsampled = method2_resample_data_with_distance_correction(
        actual_curve_slope,
        actual_curve_gaps,
        len(bspline_1_8)
    )

    
    spline_1_4_2d = -generate_new_curve(
        bspline_1_8, 
        support_curve_m1_upsampled, 
        actual_curve_slope_upsampled, 
        support_curve_d1_upsampled, 
        actual_curve_gaps_upsampled
        )
    z_bspline_1_8 = bspline_1_8[:, 1] * np.tan(np.radians(wedge_angle) / 2)
    bspline_1_8 = np.column_stack(
        (bspline_1_8[:, 0], bspline_1_8[:, 1], z_bspline_1_8)
    )
    # spline_1_4_2d[:, 1] = make_y_descending(spline_1_4_2d[:, 1])
    spline_1_4_2d = scale_curve_y_to_range(spline_1_4_2d, 0, domain_height)
    max_x_spline_1_4_2d = spline_1_4_2d.max(axis=0)[0]
    min_x_spline_1_4_2d = spline_1_4_2d.min(axis=0)[0]
    spline_1_4_2d = scale_curve_x_to_range(spline_1_4_2d, min_x_spline_1_4_2d - 1, 0)
    max_x_spline_1_4_2d = spline_1_4_2d.max(axis=0)[0]
    min_x_spline_1_4_2d = spline_1_4_2d.min(axis=0)[0]
    support_curve_14_gap = max_x_spline_1_4_2d - min_x_spline_1_4_2d

    x = spline_1_4_2d[:, 0]  # First column (x values)
    y = spline_1_4_2d[:, 1]  # Second column (y values)
    # # z = spline_14_16_2d[:, 2]  # Third column (z values)

    z_1_4 = spline_1_4_2d[:, 1] * np.tan(np.radians(wedge_angle) / 2)

    spline_1_4 = np.column_stack((spline_1_4_2d[:, 0], spline_1_4_2d[:, 1], z_1_4))


    # Create spline 1-2 by negating the z-values of spline 1-4
    z_1_2 = -spline_1_4_2d[:, 1] * np.tan(np.radians(wedge_angle) / 2)
    spline_1_2 = np.column_stack((spline_1_4_2d[:, 0], spline_1_4_2d[:, 1], z_1_2))
    
    # spline_1_4 = spline_1_4.tolist()
    spline_1_2 = spline_1_2.tolist()


    bspline_1_6['z'] = -bspline_1_6['z']  # Invert z
    bspline_1_6 = bspline_1_6.to_numpy().tolist()
    forebody_z = forebody_coords[:, 1] * np.tan(np.radians(wedge_angle) / 2)
    forebody = np.column_stack(
        (forebody_coords[:, 0], forebody_coords[:, 1], forebody_z)
    )
    
    # Step 3: Aft body (Spline 12 14 and Spline 10 14)
    print("GENERATION:::AFTBODY")
    aftbody_coords = coordinates.iloc[j:].copy()
    # spline_10_14 = aftbody_coords.copy()
    aftbody_coords['z'] = aftbody_coords['Y'] * np.tan(np.radians(wedge_angle) / 2)
    spline_12_14 = aftbody_coords[['X', 'Y']].to_numpy()

    slope_distance_df_aftbody = pd.read_csv('slopes_distances_14_16_14_12.csv')

    support_curve_slope = slope_distance_df_forebody['m1'].to_numpy()
    actual_curve_slope = slope_distance_df_forebody['m2'].to_numpy()
    support_curve_gaps = slope_distance_df_forebody['d1'].to_numpy()
    actual_curve_gaps = slope_distance_df_forebody['d2'].to_numpy()
    
    aftbody_coords = aftbody_coords.to_numpy()
    aftbody_coords = aftbody_coords[:, :2]

    # method2_resample_data_with_distance_correction is used to resample the number of points, such that we have same number of points for curve generation process
    # NOTE: Last arguement of this function here passed as len(spline_12_14) effectively defines how many points we need in the curve so this we can change to like 80 or even 4
    # Less points Less accuracy; More points More computation
    support_curve_m1_upsampled_aft, support_curve_d1_upsampled_aft = method2_resample_data_with_distance_correction(
        support_curve_slope,
        support_curve_gaps,
        len(spline_12_14)
    )
    actual_curve_slope_upsampled_aft, actual_curve_gaps_upsampled_aft = method2_resample_data_with_distance_correction(
        actual_curve_slope,
        actual_curve_gaps,
        len(spline_12_14)
    )

    if PRINT_LOGS:
        print(f'{spline_12_14.shape = }')
        print(f'{support_curve_m1_upsampled_aft.shape = }')
        print(f'{actual_curve_slope_upsampled_aft.shape = }')
        print(f'{support_curve_d1_upsampled_aft.shape = }')
        print(f'{actual_curve_gaps_upsampled_aft.shape = }')
    # This creates the actual "support" curve
    spline_14_16_2d = generate_new_curve(
        spline_12_14[::-1], 
        support_curve_m1_upsampled_aft,
        actual_curve_slope_upsampled_aft,
        support_curve_d1_upsampled_aft,
        actual_curve_gaps_upsampled_aft
        )
    
    z_spline_12_14 = spline_12_14[:, 1]* np.tan(np.radians(wedge_angle) / 2)
    spline_12_14 = np.column_stack(
        (spline_12_14[:, 0], spline_12_14[:, 1], z_spline_12_14)
    )

    # NOTE: I realised that somehow this curve is not monotous increasing only, so had to use this function; One should check the profile of y before running this function
    spline_14_16_2d[:, 1] = make_y_ascending(spline_14_16_2d[:, 1]) 
    spline_14_16_2d = scale_curve_y_to_range(spline_14_16_2d, 0, domain_height) # The curve is then rescaled from 0 to domain height
    max_spline_14_16_2d = spline_14_16_2d.max(axis=0)[0]
    min_spline_14_16_2d = spline_14_16_2d.min(axis=0)[0]
    if PRINT_LOGS: print(f"{max_spline_14_16_2d - min_spline_14_16_2d = }")

    # here the number '1' written in the third arguement defines how far awar the end point of the curve would be FOR FOREBODY BODY
    spline_14_16_2d = scale_curve_x_to_range(spline_14_16_2d, min_spline_14_16_2d, max_spline_14_16_2d + 1)
    max_spline_14_16_2d = spline_14_16_2d.max(axis=0)[0]
    min_spline_14_16_2d = spline_14_16_2d.min(axis=0)[0]
    aft_body_support_curve_x_gap = max_spline_14_16_2d - min_spline_14_16_2d
    if PRINT_LOGS: print(f'{aft_body_support_curve_x_gap = }')
    
    z_14_16 = spline_14_16_2d[:, 1] * np.tan(np.radians(wedge_angle) / 2)
    # exit()


    spline_14_16 = np.column_stack(
        (spline_14_16_2d[:, 0], spline_14_16_2d[:, 1], z_14_16)
    )
    
    z_14_15 = -spline_14_16_2d[:, 1] * np.tan(np.radians(wedge_angle) / 2) 
    spline_14_15 = np.column_stack(
        (spline_14_16_2d[:, 0], spline_14_16_2d[:, 1], z_14_15)
    )

    z_aft = aftbody_coords[:, 1] * np.tan(np.radians(wedge_angle) / 2)
    aftbody = np.column_stack(
        (aftbody_coords[:, 0], aftbody_coords[:, 1], z_aft)
    )
    
    if PLOT_SHOW:
        plt.plot(bspline_1_8[:, 0], bspline_1_8[:, 1], '-', label = '1 8 curve')
        plt.plot(spline_1_4[:, 0], spline_1_4[:, 1], '-', label = '1 4 curve')
        plt.plot(spline_12_14[:, 0], spline_12_14[:, 1], '-', label = '12 14 curve')
        plt.plot(spline_14_16[:, 0], spline_14_16[:, 1], '-', label = '14 16 curve')
        plt.legend()
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.show()

    spline_10_14 = spline_12_14.copy()
    spline_10_14 = np.column_stack(
        (spline_10_14[:, 0], spline_10_14[:, 1], -spline_10_14[:, 2])
    )
    spline_14_16 = spline_14_16.tolist()
    spline_14_15 = spline_14_15.tolist()



    # spline_10_14['z'] = -spline_10_14['z']  # Invert z
    spline_10_14 = spline_10_14.tolist()
    
    # exit()
    # Step 5: Vertices creation
    print("GENERATION ::: VERTEX ")
    vertices = [0]*20 # assuming 20 points to be there
    # The numbering of the vertices is written as described in the obd file in the same directory
    # The number starts with 0 only

    # -------------------------------------------------------------
    vertex_1 = [coordinates.iloc[0, 0], 0, 0]
    vertices[1] = vertex_1

    vertex_14 = [coordinates.iloc[-1, 0], 0, 0]
    vertices[14] = vertex_14

    vertex_8 = forebody[-1].copy().tolist()
    vertices[8] = vertex_8

    vertex_6 = vertex_8.copy()
    vertex_6[2] = -vertex_8[2]
    vertices[6] = vertex_6

    vertex_12 = aftbody[0].copy().tolist()
    vertex_12[1] = vertex_8[1]
    vertices[12] = vertex_12

    vertex_10 = vertex_12.copy()
    vertex_10[2] = -vertex_12[2]
    vertices[10] = vertex_10

    
    #--------------------------------------------------------------


    
    # a. Vertex 0: First coordinate of CSV - (domain left gap, 0, 0)
    vertex_0 = [coordinates.iloc[0, 0] - domain_left_gap, 0, 0]
    vertices[0] = vertex_0

    # b. Vertex 5: (x, domain height, domain top z)
    vertex_5 = [vertex_0[0], domain_height, domain_top_z]
    vertices[5] = vertex_5

    # c. Vertex 3: Same as Vertex 5 but with negative z
    vertex_3 = [vertex_0[0], domain_height, -domain_top_z]
    vertices[3] = vertex_3

    # d. Vertex 4: First coordinate of CSV + (0, domain height, domain top z)
    vertex_4 = [vertex_1[0] -support_curve_14_gap, domain_height, domain_top_z]
    vertices[4] = vertex_4

    # e. Vertex 2: Same as Vertex 4 but with negative z
    vertex_2 = [vertex_1[0]-support_curve_14_gap, domain_height, -domain_top_z]
    vertices[2] = vertex_2

    # f. Vertex 9: Use x from BSpline 8, (x, domain height, domain top z)
    vertex_9 = [forebody[-1, 0], domain_height, domain_top_z]
    vertices[9] = vertex_9

    # g. Vertex 7: Same as Vertex 9 but with negative z
    vertex_7 = [forebody[-1, 0], domain_height, -domain_top_z]
    vertices[7] = vertex_7

    # h. Vertex 13: Use x from Spline 12, (x, domain height, domain top z)
    vertex_13 = [aftbody[0, 0], domain_height, domain_top_z]
    vertices[13] = vertex_13

    # i. Vertex 11: Same as Vertex 13 but with negative z
    vertex_11 = [aftbody[0, 0], domain_height, -domain_top_z]
    vertices[11] = vertex_11

    # j. Vertex 16: Last coordinate of CSV + (0, domain height, domain top z)
    vertex_16 = [coordinates.iloc[-1, 0]+aft_body_support_curve_x_gap, domain_height, domain_top_z]
    vertices[16] = vertex_16

    # k. Vertex 15: Same as Vertex 16 but with negative z
    vertex_15 = [coordinates.iloc[-1, 0]+aft_body_support_curve_x_gap, domain_height, -domain_top_z]
    vertices[15] = vertex_15

    # l. Vertex 17: Last coordinate of CSV + (domain left gap, 0, 0)
    vertex_17 = [coordinates.iloc[-1, 0] + domain_left_gap, 0, 0]
    vertices[17] = vertex_17

    # m. Vertex 19: Same as Vertex 17 + (0, domain height, domain top z)
    vertex_19 = [vertex_17[0], domain_height, domain_top_z]
    vertices[19] = vertex_19

    # n. Vertex 18: Same as Vertex 19 but with negative z
    vertex_18 = [vertex_17[0], domain_height, -domain_top_z]
    vertices[18] = vertex_18

    
    for i, vertex in enumerate(vertices):
        print(f"{i}: {vertex}")

    # Step 6: Write to BlockMeshDict file using f-strings
    with open(output_file, 'w') as f:
        f.write(f"""FoamFile
        {{
            version     2.0;
            format      ascii;
            class       dictionary;
            object      blockMeshDict;
        }}
convertToMeters 1;

vertices
(""")
        # Ensure 16 decimal places in vertices
        for idx, vertex in enumerate(vertices):
            f.write(f"    ({' '.join(f'{coord:.8f}' for coord in vertex)}) //vertex {idx}\n")
        f.write(f""");

blocks
(
    hex (0 1 2 3 0 1 4 5)
    (226 380 1)
    simpleGrading
    (
        1
        (
            (30 30 1.000000)
            (170 60 3.000000)
            (1304 100 3.000000)
            (1304 100 0.333333)
            (170 60 0.333333)
            (30 30 1.000000)
        )
        1
    )
    hex (1 6 7 2 1 8 9 4)
    (160 380 1)
    simpleGrading
    (
        1
        (
            (30 30 1.000000)
            (170 60 3.000000)
            (1304 100 3.000000)
            (1304 100 0.333333)
            (170 60 0.333333)
            (30 30 1.000000)
        )
        1
    )
    hex (6 10 11 7 8 12 13 9)
    (440 380 1)
    simpleGrading
    (
        1
        (
            (30 30 1.000000)
            (170 60 3.000000)
            (1304 100 3.000000)
            (1304 100 0.333333)
            (170 60 0.333333)
            (30 30 1.000000)
        )
        1
    )
    hex (10 14 15 11 12 14 16 13)
    (160 380 1)
    simpleGrading
    (
        1
        (
            (30 30 1.000000)
            (170 60 3.000000)
            (1304 100 3.000000)
            (1304 100 0.333333)
            (170 60 0.333333)
            (30 30 1.000000)
        )
        1
    )
    hex (14 17 18 15 14 17 19 16)
    (453 380 1)
    simpleGrading
    (
        1
        (
            (30 30 1.000000)
            (170 60 3.000000)
            (1304 100 3.000000)
            (1304 100 0.333333)
            (170 60 0.333333)
            (30 30 1.000000)
        )
        1
    )
);


edges
(\n""")
        # For each bspline, correctly format and avoid semicolons after splines
        f.write(f"    BSpline 1 8\n    (\n")
        for point in bspline_1_8:
            f.write(f"        ({' '.join(f'{coord:.8f}' for coord in point)})\n")
        f.write(f"    )\n")

        f.write(f"    BSpline 1 6\n    (\n")
        for point in bspline_1_6:
            f.write(f"        ({' '.join(f'{coord:.8f}' for coord in point)})\n")
        f.write(f"    )\n")

        f.write(f"    spline 12 14\n    (\n")
        for point in spline_12_14:
            f.write(f"        ({' '.join(f'{coord:.8f}' for coord in point)})\n")
        f.write(f"    )\n")

        f.write(f"    spline 10 14\n    (\n")
        for point in spline_10_14:
            f.write(f"        ({' '.join(f'{coord:.8f}' for coord in point)})\n")
        f.write(f"    )\n")

        f.write(f"    spline 1 4\n    (\n")
        for point in spline_1_4:
            f.write(f"        ({' '.join(f'{coord:.8f}' for coord in point)})\n")
        f.write(f"    )\n")

        f.write(f"    spline 1 2\n    (\n")
        for point in spline_1_2:
            f.write(f"        ({' '.join(f'{coord:.8f}' for coord in point)})\n")
        f.write(f"    )\n")

        f.write(f"    spline 14 16\n    (\n")
        for point in spline_14_16:
            f.write(f"        ({' '.join(f'{coord:.8f}' for coord in point)})\n")
        f.write(f"    )\n")

        f.write(f"    spline 14 15\n    (\n")
        for point in spline_14_15:
            f.write(f"        ({' '.join(f'{coord:.8f}' for coord in point)})\n")
        f.write(f"    )\n")
        
        f.write(""");

boundary
(
    outlet
    {
        type patch;
        faces
        (
            (5 3 0 0)
        );
    }
    inlet
    {
        type patch;
        faces
        (
            (18 19 17 17)
        );
    }
    wedge0
    {
        type wedge;
        faces
        (
            (0 1 4 5)
            (1 8 9 4)
            (8 12 13 9)
            (12 14 16 13)
            (14 17 19 16)
        );
    }
    wedge1
    {
        type wedge;
        faces
        (
            (3 2 1 0)
            (2 7 6 1)
            (7 11 10 6)
            (11 15 14 10)
            (15 18 17 14)
        );
    }
    pipe
    {
        type wall;
        faces
        (
            (5 4 2 3)
            (4 9 7 2)
            (9 13 11 7)
            (13 16 15 11)
            (16 19 18 15)
        );
    }
    BoR
    {
        type wall;
        faces
        (
            (1 6 8 1)
            (6 10 12 8)
            (10 14 14 12)
        );
    }
);
""")

    print(f'BlockMeshDict created successfully: {output_file}')


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BlockMeshDict for a submarine using given dimensions and coordinate data.")

    parser.add_argument("--length", type=float, default=70.156, help="Length of the submarine (default: 70.156)")
    parser.add_argument("--height", type=float, default=4.799, help="Height (radius) of the submarine (default: 4.799)")
    parser.add_argument("--wedge_angle", type=float, default=5, help="Wedge angle in degrees (default: 5)")
    parser.add_argument("--coord_file", type=str, default='sf75withz.csv', help="Path to the coordinate CSV file (default: 'coordinates.csv')")
    parser.add_argument("--i", type=int, default=277, help="Index i for forebody split (default: 277)")
    parser.add_argument("--j", type=int, default=489, help="Index j for aftbody split (default: 200)")
    parser.add_argument("--output_file", type=str, default='blockMeshDict', help="Output BlockMeshDict file name (default: 'BlockMeshDict')")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # python BlockMeshDict_gen.py --length 70 --height 4 --i 267 --j 466
    create_block_mesh_dict(
        sub_length=args.length,
        sub_height=args.height,
        wedge_angle=args.wedge_angle,
        coord_file=args.coord_file,
        i=args.i,
        j=args.j,
        output_file=args.output_file
    )

