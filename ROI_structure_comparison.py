# %%
import pydicom 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from datascience import *
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString
# %%
# Define the path
rs_ref_path = "/Users/fukuyamashinpei/Documents/留学/インターン＠アルバータ/rsStructureSet/original.DCM"
rs_cmp_path = "/Users/fukuyamashinpei/Documents/留学/インターン＠アルバータ/rsStructureSet/reexport.dcm"
rs_cli_path = "/Users/fukuyamashinpei/Documents/留学/インターン＠アルバータ/rsStructureSet/clinical.dcm"

# %%
# Read the DICOM files
rs_ref = pydicom.dcmread(rs_ref_path)
rs_cmp = pydicom.dcmread(rs_cmp_path)
rs_cli = pydicom.dcmread(rs_cli_path)

# %%
# Check the structure name and the number of slices
ref_structure_name=[]
ref_slice_number=[]
cmp_structure_name=[]
cmp_slice_number=[]
cli_structure_name = []
cli_slice_number = []

# Original
for structure_num in range(len(rs_ref.StructureSetROISequence)):
    name_ref=rs_ref.StructureSetROISequence[structure_num].ROIName
    ref_structure_name.append(name_ref)
    
    slice_ref=len(rs_ref.ROIContourSequence[structure_num].ContourSequence)
    ref_slice_number.append(slice_ref)
  
# Re-export  
for structure_num in range(len(rs_cmp.StructureSetROISequence)):
    name_cmp=rs_cmp.StructureSetROISequence[structure_num].ROIName
    cmp_structure_name.append(name_cmp)
    
    slice_cmp=len(rs_cmp.ROIContourSequence[structure_num].ContourSequence)
    cmp_slice_number.append(slice_cmp)

# Clinical
for item in range(len(rs_cli.StructureSetROISequence)):
    structure = rs_cli.StructureSetROISequence[item].ROIName
    cli_structure_name.append(structure)

for structure in range(len(cli_structure_name)):
    try:
        slice_data = len(rs_cli.ROIContourSequence[structure].ContourSequence)
        cli_slice_number.append(slice_data)
    except AttributeError as e:
        cli_slice_number.append("No slice data")
    except Exception as e:
        cli_slice_number.append("No slice data")

data_ref={'Clinical Structure': cli_structure_name,
          'Slice': cli_slice_number
          }
df_cli=pd.DataFrame(data_ref)

data_ref={'Reference Structure': ref_structure_name,
          'Slice': ref_slice_number
          }
df_ref=pd.DataFrame(data_ref)

data_cmp={'Comparison Structure': cmp_structure_name,
          'Slice': cmp_slice_number
          }
df_cmp=pd.DataFrame(data_cmp)

print(df_ref)
print(df_cmp)
print(df_cli)

# %%
%matplotlib inline
def contour_comparison(rs_ref, rs_cmp_or_cli, structure, slice):
    
    structure_index_ref = ref_structure_name.index(structure)
    contour_points_ref = rs_ref.ROIContourSequence[structure_index_ref].ContourSequence[slice].ContourData
    x_coordinates_ref = np.array(contour_points_ref[0::3])
    y_coordinates_ref = np.array(contour_points_ref[1::3])
    
    if rs_cmp_or_cli is rs_cmp:
        structure_index_cmp_or_cli = cmp_structure_name.index(structure)
        contour_points_cmp_or_cli = rs_cmp_or_cli.ROIContourSequence[structure_index_cmp_or_cli].ContourSequence[
            len(rs_cmp_or_cli.ROIContourSequence[structure_index_cmp_or_cli].ContourSequence) - slice - 1].ContourData
        x_coordinates_cmp_or_cli = np.array(contour_points_cmp_or_cli[0::3])
        y_coordinates_cmp_or_cli = np.array(contour_points_cmp_or_cli[1::3])
        plt.plot(x_coordinates_ref, y_coordinates_ref, label="Original")
        plt.plot(x_coordinates_cmp_or_cli, y_coordinates_cmp_or_cli, label="Comparison")
        plt.legend()
        plt.show()
        
    else:
        structure_index_cmp_or_cli = cli_structure_name.index(structure)
        contour_points_cmp_or_cli = rs_cmp_or_cli.ROIContourSequence[structure_index_cmp_or_cli].ContourSequence[
            len(rs_cmp_or_cli.ROIContourSequence[structure_index_cmp_or_cli].ContourSequence) - slice - 1
        ].ContourData
        x_coordinates_cmp_or_cli = np.array(contour_points_cmp_or_cli[0::3])
        y_coordinates_cmp_or_cli = np.array(contour_points_cmp_or_cli[1::3])

        plt.plot(x_coordinates_ref, y_coordinates_ref, label="Original")
        plt.plot(x_coordinates_cmp_or_cli, y_coordinates_cmp_or_cli, label="Clinical")
        plt.legend()
        plt.show()
        
contour_comparison(rs_ref, rs_cli, "Kidney Right", 0)
# %%
# Calculate the Hausdorff Distance
# Original vs Re-export
structure = "Kidney Left"
hausdorff_distance_slices = []
for n_slice in range(len(rs_ref.ROIContourSequence[ref_structure_name.index(structure)].ContourSequence)):
    # Reference contour points
    contour_points_ref = rs_ref.ROIContourSequence[ref_structure_name.index(structure)].ContourSequence[n_slice].ContourData
    x_coordinates_ref = np.array(contour_points_ref[0::3])
    y_coordinates_ref = np.array(contour_points_ref[1::3])
    coordinates_ref = list(zip(x_coordinates_ref, y_coordinates_ref))  # Convert to list of tuples
    
    # Comparison contour points
    contour_points_cmp = rs_cmp.ROIContourSequence[ref_structure_name.index(structure)].ContourSequence[len(rs_ref.ROIContourSequence[ref_structure_name.index(structure)].ContourSequence) - n_slice - 1].ContourData
    x_coordinates_cmp = np.array(contour_points_cmp[0::3])
    y_coordinates_cmp = np.array(contour_points_cmp[1::3])
    coordinates_cmp = list(zip(x_coordinates_cmp, y_coordinates_cmp))  # Convert to list of tuples
    
    # Create LineString objects
    line_ref = LineString(coordinates_ref)
    line_cmp = LineString(coordinates_cmp)
    
    # Calculate Hausdorff distance
    hausdorff_distance_slices.append(line_ref.hausdorff_distance(line_cmp))

print(f"Average Hausdorff Distance: {np.average(hausdorff_distance_slices)}")
print(f"Maximum Hausdorff Distance: {np.max(hausdorff_distance_slices)}")

# %%
# Calculate Hausdorff Distance 
# Original vs Clinical
# When reference_structure_name == clinical_structure_name
matched_structures = [structure for structure in ref_structure_name if structure in cli_structure_name]
print("Matched structure: {}".format(matched_structures))

for structure in matched_structures:
    ref_index = ref_structure_name.index(structure)
    cli_index = cli_structure_name.index(structure)
    
    hausdorff_distance_slices = []
    for slice in range(ref_slice_number[ref_index]):
        contour_points_ref=rs_ref.ROIContourSequence[ref_index].ContourSequence[slice].ContourData
        x_coordinates_ref=np.array(contour_points_ref[0::3])
        y_coordinates_ref=np.array(contour_points_ref[1::3])
        coordinates_ref=list(zip(x_coordinates_ref, y_coordinates_ref))

        contour_points_cli=rs_cli.ROIContourSequence[cli_index].ContourSequence[cli_slice_number[cli_index] - slice - 1].ContourData
        x_coordinates_cli=np.array(contour_points_cli[0::3])
        y_coordinates_cli=np.array(contour_points_cli[1::3])
        coordinates_cli=list(zip(x_coordinates_cli, y_coordinates_cli))

        line_ref = LineString(coordinates_ref)
        line_cli = LineString(coordinates_cli)
        
        hausdorff_distance_slices.append(line_ref.hausdorff_distance(line_cli))
        
    print(f"Average hausdorff distance_{structure}: {np.average(hausdorff_distance_slices)}",
          f"Maximum hausdorff distance_{structure}: {np.max(hausdorff_distance_slices)}")
    
# %%
# Calculate Hausdorff Distance 
# Original vs Clinical
# Identify structures where the slice counts match and the max Hausdorff distance is small
matched_slices = [count for count in ref_slice_number if count in cli_slice_number]

ref_indices = []
for slice_count in matched_slices:
    for idx, value in enumerate(ref_slice_number):
        if value == slice_count:
            ref_indices.append(idx)

cli_indices = []
for slice_count in matched_slices:
    for idx, value in enumerate(cli_slice_number):
        if value == slice_count:
            cli_indices.append(idx)

same_slice_combination = []
for r_idx in ref_indices:
    for c_idx in cli_indices:
        if ref_slice_number[r_idx] == cli_slice_number[c_idx]:
            same_slice_combination.append([r_idx, c_idx])

max_hausdorff_distance = []
avg_hausdorff_distance = []

for ref_index, cli_index in same_slice_combination:
    hausdorff_distance_slices = []
    
    for slice_idx in range(ref_slice_number[ref_index]):
        contour_points_ref = rs_ref.ROIContourSequence[ref_index].ContourSequence[slice_idx].ContourData
        x_coordinates_ref = np.array(contour_points_ref[0::3])
        y_coordinates_ref = np.array(contour_points_ref[1::3])
        coordinates_ref = list(zip(x_coordinates_ref, y_coordinates_ref))

        contour_points_cli = rs_cli.ROIContourSequence[cli_index].ContourSequence[
            cli_slice_number[cli_index] - slice_idx - 1].ContourData
        x_coordinates_cli = np.array(contour_points_cli[0::3])
        y_coordinates_cli = np.array(contour_points_cli[1::3])
        coordinates_cli = list(zip(x_coordinates_cli, y_coordinates_cli))

        line_ref = LineString(coordinates_ref)
        line_cli = LineString(coordinates_cli)
        hausdorff_distance_slices.append(line_ref.hausdorff_distance(line_cli))
    
    max_hausdorff_distance.append(np.max(hausdorff_distance_slices))
    avg_hausdorff_distance.append(np.mean(hausdorff_distance_slices))

filtered_values_max = [val for val in max_hausdorff_distance if val <= 2]
filtered_values_avg = [val for val in avg_hausdorff_distance if val <= 1]

ref_name = []
cli_name = []

for val in filtered_values_max:
    idx = max_hausdorff_distance.index(val)
    
    pair_ref_index, pair_cli_index = same_slice_combination[idx]
    
    ref_name.append(ref_structure_name[pair_ref_index])
    cli_name.append(cli_structure_name[pair_cli_index])

data = {
    'Reference Structure': ref_name,
    'Clinical Structure': cli_name,
    'Max Hausdorff Distance': filtered_values_max,
    'Average Hausdorff distance': filtered_values_avg
}
df = pd.DataFrame(data)

print(df)