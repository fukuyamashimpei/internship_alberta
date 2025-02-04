# %%
import pydicom 
import math
import numpy as np 
from datascience import *
from scipy.spatial.transform import Rotation as R

# %%
# Define the path
rs_path="/Users/fukuyamashinpei/Documents/留学/インターン＠アルバータ/RS1.3.6.1.4.1.2452.6.3472728450.1238339071.932438716.2822116538.dcm"
rp_path="/Users/fukuyamashinpei/Documents/留学/インターン＠アルバータ/RP1.3.6.1.4.1.2452.6.1923090802.1235176203.3390519991.3034331857.dcm"

# %%
# Read the DICOM files
rs = pydicom.dcmread(rs_path)
rp = pydicom.dcmread(rp_path)

# %%
# Check the structure name
rs.StructureSetROISequence[1].ROIName

# %%
# Get the contour points of CTV
# Collect X, Y, Z coordinates in separate arrays.
x_ori_points = []
y_ori_points = []
z_ori_points = []

for n_slice in range(len(rs.ROIContourSequence[1].ContourSequence)):
    contour_points_list = rs.ROIContourSequence[1].ContourSequence[n_slice].ContourData
    
    x_ori = contour_points_list[0::3]
    y_ori = contour_points_list[1::3]
    z_ori = contour_points_list[2::3]
    
    x_ori_points = np.append(x_ori_points, x_ori)
    y_ori_points = np.append(y_ori_points, y_ori)
    z_ori_points = np.append(z_ori_points, z_ori)

# Calculate the geometric center of CTV
center_x = np.average(x_ori_points)
center_y = np.average(y_ori_points)
center_z = np.average(z_ori_points)
center = make_array(center_x, center_y, center_z)

# %%
# Translate all CTV points so that the CTV center is aligned at (0, 0, 0).
# Store the translated points in a dictionary keyed by slice number.
all_points_dict = {}
for n_slice in range(len(rs.ROIContourSequence[1].ContourSequence)):
    x_new = [x - center_x for x in x_ori_points]
    y_new = [y - center_y for y in y_ori_points]
    z_new = [z - center_z for z in z_ori_points]
    points_n = np.array([x_new, y_new, z_new]).T
    all_points_dict[n_slice] = points_n
    
# %%
catheter_points_ori = rp[(0x300F, 0x1000)][0].ROIContourSequence[2].ContourSequence[0].ContourData
point_a = np.array(catheter_points_ori[:3]) - center
point_b = np.array(catheter_points_ori[3:]) - center
xyz_prime = point_a - point_b

# Calculate the rotation angles around each axis
angle_x = math.atan2(xyz_prime[1], xyz_prime[2])
angle_y = math.atan2(xyz_prime[0], xyz_prime[2])
angle_z = math.atan2(xyz_prime[0], xyz_prime[1])

rotation = R.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=False)

# %%
# Apply the rotation to each set of contour points (translated CTV).
rotated_points_dict = {}
for slice_number, points in all_points_dict.items():
    rotated_points = rotation.apply(points)
    rotated_points_dict[slice_number] = rotated_points
    
# %%
# Collect all rotated coordinates (X, Y, Z) into single arrays
rotated_all_x = []
rotated_all_y = []
rotated_all_z = []

for slice_number, points in rotated_points_dict.items():
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rotated_all_x = np.append(rotated_all_x, x)
    rotated_all_y = np.append(rotated_all_y, y)
    rotated_all_z = np.append(rotated_all_z, z) 

width = np.max(rotated_all_x) - np.min(rotated_all_x)
height = np.max(rotated_all_y) - np.min(rotated_all_y)
thickness = np.max(rotated_all_z) - np.min(rotated_all_z)

print('Width :', width)
print('Height :', height)
print('Thickness :', thickness)

# %%
bin=5
z_slice=[]
for i in range(math.ceil((np.max(rotated_all_z) - np.min(rotated_all_z))/bin)):
    z=np.min(rotated_all_z) + bin * i
    z_slice=np.append(z_slice,z)

all_points_list=[]
for slice_number, points in rotated_points_dict.items():
    all_points_list=points

# %%
# More precise width
max_width_in_slice=[]
for i in range(math.ceil((np.max(rotated_all_z) - np.min(rotated_all_z))/bin-1)):
    bool_mask=(z_slice[i] <= all_points_list[:, 2]) & (all_points_list[:, 2] <= z_slice[i+1])
    filtered_list=all_points_list[bool_mask]
    max_width_in_slice=np.append(max_width_in_slice, np.max(filtered_list[:, 0])-np.min(filtered_list[:, 0]))
    print(np.max(filtered_list[:, 0])-np.min(filtered_list[:, 0]))
more_precise_width=np.max(max_width_in_slice)

# %%
# More precise height
max_height_in_slice=[]
for i in range(math.ceil((np.max(rotated_all_z) - np.min(rotated_all_z))/bin)-1):
    bool_mask=(z_slice[i] <= all_points_list[:, 2]) & (all_points_list[:, 2] <= z_slice[i+1])
    filtered_list=all_points_list[bool_mask]
    max_height_in_slice=np.append(max_height_in_slice, np.max(filtered_list[:, 1])-np.min(filtered_list[:, 1]))
    print(np.max(filtered_list[:, 1])-np.min(filtered_list[:, 1]))
more_precise_height=np.max(max_height_in_slice)

# %%
print('More precise width: {}'.format(more_precise_width))
print('More precise height: {}'.format(more_precise_height))

# %%
# Rotate point_a and point_b (i.e., two catheter points) to get their new positions.
a_prime = rotation.apply(point_a)
b_prime = rotation.apply(point_b)

# Compute the mean position of the rotated tandem points (i.e., the midpoint in rotated space).
mean_rotated_tandem_point = make_array(
    np.average([a_prime[0], b_prime[0]]),
    np.average([a_prime[1], b_prime[1]]),
    np.average([a_prime[2], b_prime[2]])
)

# Translate all rotated CTV points so the tandem midpoint is at the origin (0,0,0).
rotated_all_x_moved = [x - mean_rotated_tandem_point[0] for x in rotated_all_x]
rotated_all_y_moved = [y - mean_rotated_tandem_point[1] for y in rotated_all_y]
rotated_all_z_moved = [z - mean_rotated_tandem_point[2] for z in rotated_all_z]

# Compute the maximum radial distance in the X-Y plane
max_radial_distance = math.sqrt(np.max(rotated_all_x_moved)**2 + np.max(rotated_all_y_moved)**2)
print('Max radial distance: {}'.format(max_radial_distance))