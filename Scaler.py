import os
import numpy as np

def scale_point_clouds_in_room(room_directory, scaling_factor):
    """
    This function scales all point clouds in the 'Annotations' folder proportionally to the center
    of the combined point clouds, then rewrites the 'room_x.txt' file.

    Args:
        room_directory (str): The path to the room directory (room_x) containing the 'Annotations' folder.
        scaling_factor (float): The factor by which the point clouds will be scaled.
    """
    annotations_dir = os.path.join(room_directory, 'Annotations')
    if not os.path.exists(annotations_dir):
        print(f"No 'Annotations' folder found in {room_directory}")
        return

    all_points = []  # List to store all the points from the .txt files
    point_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]  # Get all .txt files

    if not point_files:
        print(f"No point cloud files found in {annotations_dir}")
        return

    # Step 1: Read all point clouds and accumulate points
    for file_name in point_files:
        file_path = os.path.join(annotations_dir, file_name)
        point_cloud = np.loadtxt(file_path)  # Load the XYZRGB data
        all_points.append(point_cloud[:, :3])  # Extract only XYZ for center calculation

    # Step 2: Combine all points and calculate the center
    all_points_combined = np.vstack(all_points)
    center_of_point_clouds = np.mean(all_points_combined, axis=0)

    # Step 3: Scale each point cloud relative to the center and rewrite the files
    for file_name in point_files:
        file_path = os.path.join(annotations_dir, file_name)
        point_cloud = np.loadtxt(file_path)  # Load the XYZRGB data
        xyz = point_cloud[:, :3]  # Extract XYZ

        # Scale XYZ relative to the center of all point clouds
        scaled_xyz = center_of_point_clouds + scaling_factor * (xyz - center_of_point_clouds)

        # Combine the scaled XYZ with RGB values
        scaled_point_cloud = np.hstack((scaled_xyz, point_cloud[:, 3:]))

        # Rewrite the file with the scaled point cloud
        np.savetxt(file_path, scaled_point_cloud, fmt='%.6f %.6f %.6f %.6f %.6f %.6f')

    # Step 4: Merge all scaled files into room_x.txt file
    output_file = os.path.join(room_directory, f'{os.path.basename(room_directory)}.txt')
    with open(output_file, 'w') as outfile:
        for file_name in point_files:
            file_path = os.path.join(annotations_dir, file_name)
            with open(file_path, 'r') as infile:
                outfile.write(infile.read())
                outfile.write('\n')  # Add a newline after each file's content

    print(f'Successfully scaled and merged files into {output_file}')

def process_area_rooms(area_directory, scaling_factor):
    """
    This function processes each room in the given area directory, scaling its point clouds.

    Args:
        area_directory (str): The path to the area directory (Area_x).
        scaling_factor (float): The scaling factor to be applied to the point clouds.
    """
    rooms = [d for d in os.listdir(area_directory) if d.startswith('room')]
    for room in rooms:
        room_directory = os.path.join(area_directory, room)
        scale_point_clouds_in_room(room_directory, scaling_factor)

def scale_dataset(dataset_directory, scaling_factor):
    """
    This function processes all areas and rooms in the dataset, scaling point clouds in each room.

    Args:
        dataset_directory (str): The root directory of the dataset containing Area_x folders.
        scaling_factor (float): The scaling factor to be applied to the point clouds.
    """
    areas = [d for d in os.listdir(dataset_directory) if d.startswith('Area')]
    for area in areas:
        area_directory = os.path.join(dataset_directory, area)
        print(f"Processing {area_directory}...")
        process_area_rooms(area_directory, scaling_factor)

if __name__ == '__main__':
    # Example usage
    dataset_directory = r'path_to_your_dataset_folder'  # Change this to your actual dataset folder
    scaling_factor = 0.3  # Example scaling factor, change as needed

    # Process the entire dataset
    scale_dataset(dataset_directory, scaling_factor)