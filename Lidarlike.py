import os
import shutil
import numpy as np
import random
import open3d as o3d
from tqdm import tqdm
import time


def copy_directory_with_files(src, dst):
    """
    Copies the entire directory structure and files from the source to the destination.

    Args:
        src (str): Source directory path.
        dst (str): Destination directory path.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    for root, dirs, files in os.walk(src):
        # Copy directories
        for directory in dirs:
            src_dir = os.path.join(root, directory)
            dst_dir = os.path.join(dst, os.path.relpath(src_dir, src))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

        # Copy files
        for file in tqdm(files, desc=f"Copying files from {root}", unit="file"):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst, os.path.relpath(src_file, src))
            shutil.copy2(src_file, dst_file)

    print(f"Copied all files and directory structure from {src} to {dst}.")


def read_xyzrgb_file(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:6] / 255.0  # Normalize colors to [0, 1] range
    return points, colors


def process_point_cloud(file_path):
    print(f"Processing {file_path}")

    # Read point cloud from XYZRGB file
    points, colors = read_xyzrgb_file(file_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Calculate the center of the point cloud
    center = pcd.get_center()

    # Estimate the diameter of the point cloud
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

    # Define parameters for hidden point removal for different diagonal camera angles
    camera_positions = [
        center + np.array([diameter, diameter, diameter]),  # Diagonal top-right view
        center + np.array([-diameter, diameter, diameter]),  # Diagonal top-left view
        center + np.array([diameter, -diameter, diameter]),  # Diagonal bottom-right view
        center + np.array([-diameter, -diameter, diameter]),  # Diagonal bottom-left view
    ]
    radius = diameter * 100

    # Select one of the visible point clouds at random to replace the original file
    selected_camera = random.choice(camera_positions)
    print(f"Selected camera position: {selected_camera}")
    _, pt_map = pcd.hidden_point_removal(selected_camera, radius)
    visible_pcd = pcd.select_by_index(pt_map)

    # Convert the visible point cloud to a NumPy array
    visible_points = np.asarray(visible_pcd.points)
    visible_colors = np.asarray(visible_pcd.colors) * 255.0  # Convert back to original color range
    visible_data = np.hstack((visible_points, visible_colors))

    # Replace the original file with the chosen point cloud
    np.savetxt(file_path, visible_data, fmt='%.6f')
    print(f"Replaced original file with new view: {file_path}")


def process_annotations_folder(folder_path):
    """
    Process all .txt files in the annotations folder and update them using the process_point_cloud function.
    """
    files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if
             file.endswith(".txt") and not any(skip in file for skip in ["ceiling", "floor", "wall", "room", "Angle"])]

    for file_path in tqdm(files, desc=f"Processing point clouds in {folder_path}", unit="file"):
        process_point_cloud(file_path)


def consolidate_point_clouds(directory, output_file):
    """
    Consolidates all point clouds from text files in the specified directory into a single file.

    Args:
        directory (str): Path to the directory containing the text files with point cloud data.
        output_file (str): Name of the output file to store the consolidated point cloud data.
    """
    output_path = os.path.join(directory, output_file)
    with open(output_path, 'w') as outfile:
        files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if
                 file.endswith('.txt')]

        for file_path in tqdm(files, desc=f"Consolidating point clouds in {directory}", unit="file"):
            try:
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read())  # Write all lines from the input file
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    print(f"Consolidated point cloud data saved to {output_path}")


def process_rooms_in_areas(dest_directory):
    """
    Iterates over the Areas folders, consolidates point clouds from each room's annotations folder
    into a single .txt file named exactly as the room folder after files have been copied.

    Args:
        dest_directory (str): Path to the destination directory containing the copied Areas folders.
    """
    for area_root, area_dirs, _ in tqdm(os.walk(dest_directory), desc="Processing Areas", unit="area"):
        for area_dir in area_dirs:
            area_path = os.path.join(area_root, area_dir)
            for room_root, room_dirs, _ in tqdm(os.walk(area_path), desc=f"Processing rooms in {area_dir}",
                                                unit="room"):
                for room_dir in room_dirs:
                    room_path = os.path.join(room_root, room_dir)
                    annotations_path = os.path.join(room_path, 'annotations')

                    if os.path.exists(annotations_path):
                        # Process and replace the point clouds
                        process_annotations_folder(annotations_path)

                        # Generate the output filename exactly matching the room folder name
                        room_name = os.path.basename(room_path)
                        output_file = f"{room_name}.txt"
                        consolidate_point_clouds(room_path, output_file)

'''
# Example usage
if __name__ == "__main__":
    dirs = os.listdir(r"E:\PycharmProjects\LowPython\Datasets")
    print(dirs)
    for dir in dirs:
        src_directory_path = f"E:\\PycharmProjects\\LowPython\\Datasets\\{dir}"
        dest_directory_path = f"E:\\PycharmProjects\\LowPython\\Datasets\\{dir}_lidar"

        start_time = time.time()
        # Step 1: Copy all files and directory structure to the destination
        copy_directory_with_files(src_directory_path, dest_directory_path)
        print(f"Time taken to copy files: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        # Step 2: Process files in the destination directory
        process_rooms_in_areas(dest_directory_path)
        print(f"Time taken to process point clouds: {time.time() - start_time:.2f} seconds")
'''
