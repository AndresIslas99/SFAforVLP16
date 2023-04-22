import os
import numpy as np
import laspy

def kitti_to_las(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    las = laspy.create(file_version="1.2", point_format=2)
    las.header.offset = [0, 0, 0]
    las.header.scale = [0.001, 0.001, 0.001]
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.intensity = points[:, 3]
    return las

def downsample_to_vlp16(las):
    azimuths = np.arctan2(las.y, las.x)  # Compute azimuth angles of the points
    sorted_indices = np.argsort(azimuths)  # Sort points based on azimuth angles
    sorted_las_points = las.points[sorted_indices]
    vlp16_ratio = 16 / 64  # We are downsampling from 64 channels to 16 channels
    selected_indices = np.floor(np.linspace(0, len(sorted_las_points)-1, int(len(sorted_las_points) * vlp16_ratio))).astype(int)
    downsampled_points = sorted_las_points[selected_indices]
    las.points = downsampled_points
    return las

def save_vlp16_to_kitti(las, output_path):
    points = np.column_stack((las.x, las.y, las.z, las.intensity))
    points.astype(np.float32).tofile(output_path)


input_folder = '/home/andres/Documents/SFA3D/SFA3D/dataset/kitti/testing/velodyne'
output_folder = '/home/andres/Documents/SFA3D/SFA3D/dataset/kitti/testing/vlp16'
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    las = kitti_to_las(input_path)
    las_vlp16 = downsample_to_vlp16(las)
    save_vlp16_to_kitti(las_vlp16, output_path)
