
# Procedural-PointCloudForge

## Overview

Procedural-PointCloudForge is a Python-based tool designed to generate synthetic point cloud datasets of 3D scenes with procedurally placed objects. It allows users to create realistic point cloud data for testing and training purposes, which is particularly useful in fields such as computer vision, robotics, and machine learning.

The tool leverages procedural generation techniques to create diverse and customizable 3D environments, including various objects like walls, floors, ceilings, pipes, and more. Users can specify parameters such as point density and noise levels to simulate different sensing conditions.

## Features

- **Procedural Generation**: Create 3D scenes with procedurally placed objects based on a customizable scene graph.
- **Customizable Objects**: Support for various 3D objects including walls, floors, ceilings, doors, pipes, and more.
- **Point Cloud Sampling**: Generate point clouds using Poisson disk sampling with specified point densities.
- **Noise Simulation**: Add customizable Gaussian noise to simulate sensor inaccuracies.
- **Dataset Structuring**: Save generated scenes and annotations in a structured directory format compatible with common dataset standards.
- **Visualization Tools**: Visualize generated 3D scenes and scene graphs using Open3D and Matplotlib.
- **Scaling and Renaming**: Utilities to scale datasets and rename files/folders to match specific naming conventions.

## Dependencies

- Python 3.x
- [Open3D](http://www.open3d.org/) (`open3d`)
- NumPy (`numpy`)
- NetworkX (`networkx`)
- Matplotlib (`matplotlib`)
- tqdm (`tqdm`)

Additional custom modules (included in the repository):

- `Lidarlike.py`
- `Scaler.py`
- `BasicRenamer.py`

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/Procedural-PointCloudForge.git
   cd Procedural-PointCloudForge
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Required Python Packages**

   ```bash
   pip install -r requirements.txt
   ```

   If a `requirements.txt` file is not provided, install dependencies manually:

   ```bash
   pip install open3d numpy networkx matplotlib tqdm
   ```

4. **Prepare Additional Files**

   - **Constraints File**: Ensure that the `constraints.json` file is present in the root directory. This file defines the scene graph and object constraints.
   - **Object Meshes and Orientations**: Place the required object mesh files and orientation data in the appropriate directories as expected by the script. Refer to the [Customization](#customization) section for details.

## Usage

The main script is designed to generate point cloud datasets based on specified parameters. You can modify the parameters in the `__main__` section of the script to customize the dataset generation.

### Parameters

- **`number_of_areas`**: The number of different area datasets to generate.
- **`number_of_rooms`**: The number of rooms to generate within each area.
- **`point_densities`**: A list of point densities to use when sampling point clouds.
- **`point_noises`**: A list of noise levels to apply to the point clouds.
- **`dataset_main`**: The root directory where datasets will be saved.
- **`test_mode`**: If set to `True`, the script will display visualizations for testing purposes.

### Running the Script

To run the script, execute:

```bash
python procedural_pointcloud_forge.py
```

Replace `procedural_pointcloud_forge.py` with the actual filename if different.

### Example Configuration

In the `__main__` section, you can customize the parameters:

```python
if __name__ == '__main__':
    number_of_areas = 10
    number_of_rooms = 45
    test_mode = False  # Set to True to enable visualization
    dataset_main = r'/path/to/your/dataset/root'

    point_densities = [10000]
    point_noises = [5]
```

This configuration will generate 10 areas, each with 45 rooms, using a point density of 10,000 and a noise level of 5%.

### Output Structure

The generated datasets will be saved in the specified `dataset_main` directory, organized as:

```
dataset_main/
    Datasets_<noise>_<density>_final/
        Area_1/
            room_1/
                Annotations/
                    object_name_1.txt
                    object_name_2.txt
                    ...
                room_1.txt  # Merged point cloud for the room
            room_2/
            ...
        Area_2/
        ...
```

Each `Annotations` folder contains point cloud files for individual objects in the scene, and a merged file for the entire room.

## Visualization

If `test_mode` is set to `True`, the script will display visualizations of the generated 3D scenes and scene graphs.

- **3D Scene Visualization**: Uses Open3D to render the generated point clouds.
- **Scene Graph Visualization**: Uses Matplotlib and NetworkX to display the scene graph.

## Available Datasets

Refer to the [Dataset Generator](https://github.com/danielflopez1/Procedural-PointCloudForge/tree/main) additionally the dataset can be downloaded [here](https://drive.google.com/drive/folders/1Mm_oXaCFNzYMCjW9fn5Y-9vCxJXCqb3N?usp=drive_link)

## Customization

You can customize the scene generation by modifying the `constraints.json` file and by providing your own object meshes and orientation data.

### Modifying the Scene Graph

The `constraints.json` file defines the objects in the scene and their relationships. You can edit this file to:

- Add new object types.
- Define spawning conditions and constraints.
- Specify object sizes and instances.

### Adding New Objects

To add new objects:

1. **Define the Object in `constraints.json`**

   Add a new entry for the object with its attributes, such as size, conditions, and spawning links.

   ```json
   {
     "NewObject": {
       "Size": [width, height, depth],
       "Instances": number_of_instances,
       "Conditions": {
         "Grounding": "Floor",
         "DistanceFromObjects": min_distance
       },
       "SpawningLink": ["Floor", "Wall"]
     },
     ...
   }
   ```

2. **Provide Mesh Files**

   Place the mesh files for the object in the appropriate directory structure, for example:

   ```
   /path/to/meshes/NewObject/
       NewObject_1.pcd
       NewObject_2.pcd
       ...
   ```

3. **Orientation Data**

   If the object requires specific orientations, provide an orientations file:

   ```
   /path/to/orientations/NewObject.txt
   ```

   Each line in the file should specify an orientation in degrees (e.g., `0,90,0`).

### Adjusting Noise and Density

- **Point Density**: Adjust the `point_densities` list to include desired point densities. Higher values result in denser point clouds.
- **Point Noise**: Adjust the `point_noises` list to include desired noise levels (in percentage). This simulates sensor inaccuracies.

### Changing Paths

Update paths in the script to point to your data directories. For example:

```python
dataset_main = r'/path/to/your/dataset/root'
object_path = r'/path/to/your/object/data'
```

## Utilities

The script includes utilities for:

- **Merging Annotation Files**: Merges individual object point cloud files into a single file per room.
- **Scaling Datasets**: Scales the point cloud data to simulate different sensor resolutions or scene sizes.
- **Renaming Files and Folders**: Renames files and folders to match specific naming conventions or standards.

### Merging Annotation Files

The function `merge_txt_files_in_annotations(base_directory)` combines individual object `.txt` files into a single file for each room.

### Scaling Datasets

Use `scale_dataset(base_directory, scale)` to scale the point cloud coordinates by a specified factor.

### Renaming Files and Folders

The `rename_files_and_folders(base_directory, original_name, new_name)` function renames files and folders to match a new naming convention.

## Troubleshooting

- **ModuleNotFoundError**: Ensure that custom modules like `Lidarlike`, `Scaler`, and `BasicRenamer` are in your Python path or the same directory as the main script.
- **FileNotFoundError**: Verify that all required files, such as `constraints.json` and mesh files, are in the correct locations.
- **Visualization Issues**: If visualizations are not displaying, check that `test_mode` is set to `True` and that you have a GUI environment available.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue or contact the project maintainer at [dflopezm@uwaterloo.ca](mailto:dflopezm@uwaterloo.ca).

---

**Note**: Replace `/path/to/your/dataset/root`, `/path/to/your/object/data` with the appropriate paths specific to your setup.
