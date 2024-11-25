import open3d as o3d  # For 3D visualization and mesh manipulation
import numpy as np  # For numerical operations
import json  # For handling JSON files
import networkx as nx  # For creating and manipulating graphs
import matplotlib.pyplot as plt  # For plotting graphs
import random  # For random number generation
from random import randrange  # For generating random numbers within a range
import os  # For file operations
from tqdm import tqdm
from Lidarlike import copy_directory_with_files, process_rooms_in_areas
from Scaler import scale_dataset
from BasicRenamer import rename_files_and_folders
random.seed(45)
np.random.seed(45)
basic_pcds = {}

def sample_point_cloud_poisson(mesh, base_density=5000, max_points=10000):
    """Sample a point cloud from the mesh using Poisson disk sampling."""
    surface_area = mesh.get_surface_area()
    desired_num_points = int(base_density * surface_area)
    desired_num_points = min(desired_num_points, max_points)
    # print(f"Desired Number of Points (Poisson Sampling): {desired_num_points}")
    pcd = mesh.sample_points_poisson_disk(number_of_points=desired_num_points)
    # print(f"Sampled {len(pcd.points)} points using Poisson disk sampling.")
    return pcd



def create_alignment_files(base_directory):
    for root, dirs, files in os.walk(base_directory):
        area_folder = os.path.basename(root)

        # Check if there are any folders named 'room' in the current directory
        rooms = [d for d in dirs if d.startswith('room')]
        if rooms:
            area_index = int(area_folder.split('_')[-1]) - 1  # Extract the area index from the folder name

            # Alignment angle file path
            alignment_file_path = os.path.join(root, f'Area_{area_index + 1}_alignmentAngle.txt')

            # Create alignment angles and write them to the file
            with open(alignment_file_path, 'w') as f:
                f.write(f"## Global alignment angle per disjoint space in Area_{area_index + 1} ##\n")
                f.write("## Disjoint Space Name Global Alignment Angle ##\n")

                # Loop through the rooms and assign a random alignment angle
                for room_index, room in enumerate(range(number_of_rooms)):
                    alignment_angle = random.choice([0, 90, 180, 270])
                    f.write(f"room_{room + 1} {alignment_angle}\n")

            print(f"Created alignment file: {alignment_file_path}")


def merge_txt_files_in_annotations(base_directory):
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(base_directory):
        # Check if current directory has an Annotations folder
        if 'Annotations' in dirs:
            annotations_path = os.path.join(root, 'Annotations')
            parent_folder_name = os.path.basename(root)
            output_file = os.path.join(root, f'{parent_folder_name}.txt')  # Save outside Annotations

            # Collect and merge all .txt files from Annotations folder
            with open(output_file, 'w') as outfile:
                for file_name in os.listdir(annotations_path):
                    if file_name.endswith('.txt'):
                        file_path = os.path.join(annotations_path, file_name)
                        with open(file_path, 'r') as infile:
                            outfile.write(infile.read())
                            outfile.write('\n')  # Add a newline after each file's content
            print(f'Merged files from {annotations_path} into {output_file}')


class SceneGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Initialize a directed graph using NetworkX

    def parse_constraints(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)  # Load the JSON data from the file

        for node, attributes in data.items():
            self.graph.add_node(node, **attributes)
            for link in attributes.get("SpawningLink", []):
                self.graph.add_edge(link, node)


class ObjectPlacer:
    def __init__(self, scene_graph, room_size, visualizer, noise, density):
        self.scene_graph = scene_graph
        self.room_size = room_size
        self.visualizer = visualizer
        self.helper = PlacementHelper(room_size, scene_graph)
        self.placed_objects = {}
        self.color_map = {}
        self.point_noise = noise / 100
        self.point_density = density

    def generate_unique_colors(self):
        """
        Generate a unique color map for the scene.
        """
        object_types = list(self.scene_graph.graph.nodes)
        self.color_map = {obj: [random.random(), random.random(), random.random()] for obj in object_types}

    def ensure_object_list(self, object_name):
        if object_name not in self.placed_objects:
            self.placed_objects[object_name] = []

    def place_floor(self):
        object_name = "Floor"
        color = self.color_map[object_name]
        object_attributes = self.scene_graph.graph.nodes[object_name]
        object_size = (tuple(object_attributes["Size"])[0], tuple(object_attributes["Size"])[1], 0)
        position = (0, 0, 0)
        self.ensure_object_list(object_name)

        # Create a flat box (rectangle) as the floor
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=object_size[0], height=object_size[1], depth=0.01)
        mesh_box.translate(position)  # Move the mesh to the correct position

        # Sample points from the mesh box using Poisson disk sampling
        if 'Floor' not in basic_pcds.keys():
            pcd = sample_point_cloud_poisson(mesh_box, 10000, self.point_density+10000)
            dark_gray_color = np.full((np.asarray(pcd.points).shape[0], 3), fill_value=0.2)
            pcd.colors = o3d.utility.Vector3dVector(dark_gray_color)
        else:
            pcd = basic_pcds['Floor']
        # Add the sampled point cloud to the visual scene
        self.visualizer.visual_scene.append(pcd)
        self.visualizer.save_object(pcd, object_name)
        self.placed_objects[object_name].append((position, object_size, pcd))

    def place_ceiling(self):
        object_name = "Floor"
        color = self.color_map[object_name]
        object_attributes = self.scene_graph.graph.nodes[object_name]
        ceiling_size = (tuple(object_attributes["Size"])[0], tuple(object_attributes["Size"])[1], 0)
        wall_size = tuple(self.scene_graph.graph.nodes["Wall"]["Size"])
        position = (0, 0, wall_size[2])
        self.ensure_object_list(object_name)

        # Create a flat box (rectangle) as the ceiling
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=ceiling_size[0], height=ceiling_size[1], depth=0.01)
        mesh_box.translate(position)  # Move the mesh to the correct position
        if 'ceiling' not in basic_pcds.keys():
            pcd = sample_point_cloud_poisson(mesh_box, 10000, self.point_density+10000)
            dark_gray_color = np.full((np.asarray(pcd.points).shape[0], 3), fill_value=0.2)
            pcd.colors = o3d.utility.Vector3dVector(dark_gray_color)
            basic_pcds['ceiling'] = pcd
        else:
            pcd = basic_pcds['ceiling']

        # Add the sampled point cloud to the visual scene
        self.visualizer.visual_scene.append(pcd)
        self.visualizer.save_object(pcd, object_name)
        self.placed_objects[object_name].append((position, ceiling_size, pcd))

    def place_walls(self):
        object_name = "Wall"
        wall_attributes = self.scene_graph.graph.nodes[object_name]
        floor_attributes = self.scene_graph.graph.nodes["Floor"]

        wall_thickness = wall_attributes["Size"][0]
        wall_height = wall_attributes["Size"][2]  # This should match the z-position of the ceiling
        floor_length, floor_width, _ = tuple(floor_attributes["Size"])

        self.ensure_object_list(object_name)

        # Define the four walls with correct positions and sizes
        wall_definitions = [
            # Left wall
            ((0, 0, 0), (wall_thickness, floor_width, wall_height)),
            # Right wall
            ((floor_length - wall_thickness, 0, 0), (wall_thickness, floor_width, wall_height)),
            # Front wall
            ((0, 0, 0), (floor_length, wall_thickness, wall_height)),
            # Back wall
            ((0, floor_width - wall_thickness, 0), (floor_length, wall_thickness, wall_height))
        ]

        for i, (position, size) in enumerate(wall_definitions):
            # Create a wall mesh
            mesh_wall = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
            mesh_wall.translate(position)

            # Sample points from the wall mesh using Poisson disk sampling
            pcd = sample_point_cloud_poisson(mesh_wall, base_density=self.point_density, max_points=10000)

            # Color the point cloud (white in this case)
            white_color = np.ones((np.asarray(pcd.points).shape[0], 3))
            pcd.colors = o3d.utility.Vector3dVector(white_color)

            # Add the sampled point cloud to the visual scene
            self.visualizer.visual_scene.append(pcd)

            # Save the wall with a distinct filename

            self.visualizer.save_object(pcd, object_name)

            self.placed_objects[object_name].append((position, size, pcd))

    def place_walls2(self):
        object_name = "Wall"
        object_attributes = self.scene_graph.graph.nodes[object_name]
        wall_size = tuple(object_attributes["Size"])  # (thickness, height, length)
        floor_size = tuple(self.scene_graph.graph.nodes["Floor"]["Size"])  # (length, width)
        self.ensure_object_list(object_name)

        # Define the size and position for a single encompassing wall box
        # Assuming walls surround the floor, we'll create a box that covers all walls
        # The box will have thickness equal to wall_size[0], height wall_size[1], and length as needed

        # Calculate overall dimensions
        overall_length = floor_size[0]
        overall_width = floor_size[0]
        thickness = wall_size[0]
        height = wall_size[2]

        # Create a single box that encompasses all four walls
        # The box dimensions will cover the perimeter of the floor
        # We'll set the depth to the wall thickness

        mesh_box = o3d.geometry.TriangleMesh.create_box(wall_size[1], floor_size[1], wall_size[2])
        mesh_box.compute_vertex_normals()

        # Position the box so that the inner area aligns with the floor
        # Translate the box by (-thickness, 0, -thickness)
        mesh_box.translate((-thickness, 0.0, -thickness))

        # Remove the top and bottom faces
        # To do this, we'll identify and exclude triangles with normals in +Z and -Z directions

        # Get the triangles and their normals
        triangles = np.asarray(mesh_box.triangles)
        triangle_normals = np.asarray(mesh_box.triangle_normals)

        # Define a small epsilon for numerical precision
        epsilon = 1e-6

        # Create masks for top and bottom faces
        top_face_mask = np.abs(triangle_normals[:, 2] - 1.0) < epsilon
        bottom_face_mask = np.abs(triangle_normals[:, 2] + 1.0) < epsilon

        # Combine masks to identify top and bottom triangles
        top_bottom_mask = top_face_mask | bottom_face_mask

        # Select triangles that are NOT top or bottom
        side_triangles = triangles[~top_bottom_mask]

        # Function to create a mesh from selected triangles
        def create_side_mesh(original_mesh, selected_triangles):
            side_mesh = o3d.geometry.TriangleMesh()
            # Get unique vertex indices
            unique_indices = np.unique(selected_triangles)
            # Map old indices to new
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
            # Create vertices
            vertices = np.asarray(original_mesh.vertices)[unique_indices]
            side_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            # Remap triangle indices
            triangles_mapped = [[index_map[idx] for idx in tri] for tri in selected_triangles]
            side_mesh.triangles = o3d.utility.Vector3iVector(triangles_mapped)
            side_mesh.compute_vertex_normals()
            return side_mesh

        # Create the side mesh without top and bottom
        side_mesh = create_side_mesh(mesh_box, side_triangles)

        # Optionally, assign a color to the walls (e.g., white)
        wall_color = [1.0, 1.0, 1.0]  # White
        num_vertices = len(side_mesh.vertices)
        side_mesh.paint_uniform_color(wall_color)

        # Sample points from the side mesh using Poisson disk sampling
        # Assuming `sample_point_cloud_poisson` is a predefined method
        pcd = sample_point_cloud_poisson(side_mesh, base_density=self.point_density, max_points=10000)

        # If you wish to color the point cloud differently, you can modify the colors here
        # For example, keeping them white as above
        white_color = np.ones((np.asarray(pcd.points).shape[0], 3))
        pcd.colors = o3d.utility.Vector3dVector(white_color)

        # Add the sampled point cloud to the visual scene
        self.visualizer.visual_scene.append(pcd)
        self.visualizer.save_object(pcd, object_name)
        self.placed_objects[object_name].append(((-thickness, 0.0, -thickness),(wall_size[1], floor_size[1], wall_size[2]),pcd))

    def place_walls1(self):
        object_name = "Wall"
        object_attributes = self.scene_graph.graph.nodes[object_name]
        wall_size = tuple(object_attributes["Size"])  # (thickness, height, length)
        floor_size = tuple(self.scene_graph.graph.nodes["Floor"]["Size"])  # (length, width)
        self.ensure_object_list(object_name)

        wall_positions = [
            ((0, 0, 0), (wall_size[1], floor_size[1], wall_size[2])),
            ((floor_size[0] - wall_size[1], 0, 0), (wall_size[1], floor_size[1], wall_size[2])),
            ((0, 0, 0), (floor_size[0], wall_size[1], wall_size[2])),
            ((0, floor_size[1] - wall_size[1], 0), (floor_size[0], wall_size[1], wall_size[2]))
        ]

        for pos, size in wall_positions:
            # Create a flat box (rectangle) for each wall with a small depth (thickness of the wall)
            mesh_box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
            mesh_box.translate(pos)  # Move the mesh to the correct position

            # Sample points from the mesh box using Poisson disk sampling
            pcd = sample_point_cloud_poisson(mesh_box, 500, self.point_density)
            white_color = np.ones((np.asarray(pcd.points).shape[0], 3))
            pcd.colors = o3d.utility.Vector3dVector(white_color)

            # Add the sampled point cloud to the visual scene

            self.visualizer.visual_scene.append(pcd)
            self.visualizer.save_object(pcd, object_name)
            self.placed_objects[object_name].append((pos, size, pcd))

    def place_objects(self, object_name):
        self.ensure_object_list(object_name)
        object_attributes = self.scene_graph.graph.nodes[object_name]
        size = tuple(object_attributes["Size"])
        instances = object_attributes.get('Instances', 1)
        color = self.color_map[object_name]
        spawning_links = object_attributes.get("SpawningLink", [])

        for _ in range(instances):
            potential_positions = [
                (x, y, 0) for x in range(self.room_size[0] - size[0] + 1)
                for y in range(self.room_size[1] - size[1] + 1)
            ]
            valid_positions = self.helper.check_grounding(potential_positions, object_name, size)

            if spawning_links:
                valid_positions = self.adjust_positions_near_spawning_links(valid_positions, spawning_links, size)

            valid_positions = [pos for pos in valid_positions if not self.helper.check_overlap(pos, size, 0)]

            if valid_positions:
                chosen_position = random.choice(valid_positions)
                wireframe = self.visualizer.create_wireframe_box(object_name, size, color, translate=chosen_position)
                self.placed_objects[object_name].append((chosen_position, size, wireframe))
                self.helper.mark_grid(chosen_position, size)
            else:
                break

    def adjust_positions_near_spawning_links(self, potential_positions, spawning_links, size):
        adjusted_positions = []
        for link in spawning_links:
            if link.lower() in ['floor', 'walls', 'ceiling']:
                continue
            link_positions = [pos for pos, _, _ in self.placed_objects.get(link, [])]
            link_size = tuple(self.scene_graph.graph.nodes[link]["Size"])

            for (lx, ly, _) in link_positions:
                for y in range(ly, ly + link_size[1]):
                    adjusted_positions.append((lx - size[0], y, 0))
                    adjusted_positions.append((lx + link_size[0], y, 0))
                for x in range(lx, lx + link_size[0]):
                    adjusted_positions.append((x, ly - size[1], 0))
                    adjusted_positions.append((x, ly + link_size[1], 0))

            adjusted_positions = list(set(adjusted_positions))

        return adjusted_positions if adjusted_positions else potential_positions

    def place_pipes_and_valve(self):
        object_name = "Pipe"
        self.ensure_object_list(object_name)
        support_positions = [pos for pos, _, _ in self.placed_objects.get("Support", [])]
        if not support_positions:
            return

        default_size = (1, 1, 1)
        object_attributes = self.scene_graph.graph.nodes.get(object_name, {})
        size = tuple(object_attributes.get("Size", default_size))

        path_points = self.generate_pipe_path(support_positions)
        radius_random = random.uniform(0.01, 0.7)
        if path_points is not None and not self.helper.check_pipe_overlap(path_points, radius=radius_random):
            tube, helix_wire = self.create_tube_from_path(path_points, radius=radius_random)
            wireframe = self.create_wireframe_for_tube(path_points, size)
            self.placed_objects[object_name].append((path_points, size, tube, wireframe, helix_wire))
            self.visualizer.add_tube_to_scene(tube)
            # self.visualizer.add_wireframe_to_scene(wireframe)
            # self.visualizer.add_tube_to_scene(helix_wire)
            self.helper.mark_grid_for_pipe(path_points, radius=radius_random)

            self.place_valve_on_pipe(path_points, radius_random)
        else:
            print("Failed to place pipe due to overlap.")

    def save_temporary_mesh(self, mesh, filename):
        o3d.io.write_triangle_mesh(filename, mesh)

    def save_temporary_line_set(self, line_set, filename):
        o3d.io.write_line_set(filename, line_set)

    def create_tube_from_path(self, path_points, radius, segments=20):
        vertices = []
        faces = []
        colors = []
        n = len(path_points)

        pipe_color = [random.uniform(0, 1) for _ in range(3)]
        while all(c > 0.9 for c in pipe_color):
            pipe_color = [random.uniform(0, 1) for _ in range(3)]

        for i in range(n):
            point = path_points[i]
            if i == 0:
                next_point = path_points[i + 1]
                direction = next_point - point
            elif i == n - 1:
                prev_point = path_points[i - 1]
                direction = point - prev_point
            else:
                prev_point = path_points[i - 1]
                next_point = path_points[i + 1]
                direction = next_point - prev_point

            norm_direction = np.linalg.norm(direction)
            if norm_direction > 0:
                direction /= norm_direction
            else:
                direction = np.array([1, 0, 0], dtype=float)

            if np.abs(direction[0]) > np.abs(direction[2]):
                ortho = np.array([-direction[1], direction[0], 0], dtype=float)
            else:
                ortho = np.array([0, -direction[2], direction[1]], dtype=float)

            ortho /= np.linalg.norm(ortho)
            bitangent = np.cross(direction, ortho)

            circle = []
            for j in range(segments):
                angle = 2 * np.pi * j / segments
                circle_point = (radius * np.cos(angle) * ortho +
                                radius * np.sin(angle) * bitangent)
                circle.append(point + circle_point)
                colors.append(pipe_color)

            vertices.extend(circle)

            if i > 0:
                start_idx = (i - 1) * segments
                for j in range(segments):
                    next_j = (j + 1) % segments
                    faces.append([start_idx + j, start_idx + next_j, start_idx + j + segments])
                    faces.append([start_idx + next_j, start_idx + next_j + segments, start_idx + j + segments])

        vertices = np.array(vertices)
        faces = np.array(faces)
        colors = np.array(colors)

        tube_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                             triangles=o3d.utility.Vector3iVector(faces))
        tube_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)

        helix_wire = self.create_helix_wire(path_points, radius, 0.05, turns=20)

        return tube_o3d, helix_wire

    def create_helix_wire(self, path_points, pipe_radius, wire_radius, turns=20, segments_per_turn=20):
        helix_vertices = []
        wire_radius_total = pipe_radius + wire_radius

        wire_color = [random.uniform(0, 1) for _ in range(3)]
        while all(c > 0.9 for c in wire_color):
            wire_color = [random.uniform(0, 1) for _ in range(3)]

        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            segment_length = np.linalg.norm(end_point - start_point)
            num_segments = int(segment_length * turns)

            direction = end_point - start_point

            direction = direction / (np.linalg.norm(direction) + 1e-6)

            if np.abs(direction[0]) > np.abs(direction[2]):
                ortho = np.array([-direction[1], direction[0], 0], dtype=float)
            else:
                ortho = np.array([0, -direction[2], direction[1]], dtype=float)

            ortho /= (np.linalg.norm(ortho) + 1e-6)
            bitangent = np.cross(direction, ortho)

            for j in range(num_segments):
                t = j / segments_per_turn * 2 * np.pi
                x_offset = wire_radius_total * np.cos(t)
                y_offset = wire_radius_total * np.sin(t)
                offset = x_offset * ortho + y_offset * bitangent
                point = start_point + j / num_segments * (end_point - start_point) + offset
                helix_vertices.append(point)

        t = 2 * np.pi * turns
        x_offset = wire_radius_total * np.cos(t)
        y_offset = wire_radius_total * np.sin(t)
        offset = x_offset * ortho + y_offset * bitangent
        helix_vertices.append(end_point + offset)

        helix_vertices = np.array(helix_vertices)
        lines = [[i, i + 1] for i in range(len(helix_vertices) - 1)]
        colors = [wire_color for _ in range(len(lines))]

        helix_wire = o3d.geometry.LineSet()
        helix_wire.points = o3d.utility.Vector3dVector(helix_vertices)
        helix_wire.lines = o3d.utility.Vector2iVector(lines)
        helix_wire.colors = o3d.utility.Vector3dVector(colors)

        return helix_wire

    def generate_pipe_path(self, support_positions, num_points=100, length=10):
        path_points = []
        ceiling_height = self.room_size[2] - 1

        start_wall = np.random.choice(['x', 'y'])
        if start_wall == 'x':
            start_point = np.array([0, np.random.uniform(0.5, self.room_size[1] - 0.5), ceiling_height])
            initial_turn = np.array([0.5, start_point[1], ceiling_height])
        else:
            start_point = np.array([np.random.uniform(0.5, self.room_size[0] - 0.5), 0, ceiling_height])
            initial_turn = np.array([start_point[0], 0.5, ceiling_height])

        path_points.append(start_point)
        path_points.append(initial_turn)

        current_point = initial_turn
        for support_pos in support_positions:
            next_point = np.array([support_pos[0], support_pos[1], ceiling_height])
            if current_point[0] != next_point[0]:
                intermediate_point = np.array([next_point[0], current_point[1], ceiling_height])
                path_points.append(intermediate_point)
            path_points.append(next_point)
            current_point = next_point

        end_wall_choice = np.random.choice(['perpendicular', 'parallel'])
        if end_wall_choice == 'perpendicular':
            end_wall = 'y' if start_wall == 'x' else 'x'
        else:
            end_wall = start_wall

        if end_wall == 'x':
            end_point = np.array([self.room_size[0], np.random.uniform(0.5, self.room_size[1] - 0.5), ceiling_height])
            final_turn = np.array([self.room_size[0] - 0.5, end_point[1], ceiling_height])
        else:
            end_point = np.array([np.random.uniform(0.5, self.room_size[0] - 0.5), self.room_size[1], ceiling_height])
            final_turn = np.array([end_point[0], self.room_size[1] - 0.5, ceiling_height])

        if current_point[0] != final_turn[0]:
            intermediate_point = np.array([final_turn[0], current_point[1], ceiling_height])
            path_points.append(intermediate_point)

        path_points.append(final_turn)
        path_points.append(end_point)

        path_points = np.array(path_points)

        fine_points = []
        for i in range(len(path_points) - 1):
            segment_points = np.linspace(path_points[i], path_points[i + 1], num_points // (len(path_points) - 1))
            fine_points.append(segment_points)

        spline_points = np.vstack(fine_points)
        return spline_points

    def create_wireframe_for_tube(self, path_points, size):
        lines = []
        points = []
        color = [0.0, 1.0, 0.0]

        radius = 0.5
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            box1 = [
                [p1[0] - radius, p1[1] - radius, p1[2] - radius],
                [p1[0] + radius, p1[1] - radius, p1[2] - radius],
                [p1[0] + radius, p1[1] + radius, p1[2] - radius],
                [p1[0] - radius, p1[1] + radius, p1[2] - radius],
                [p1[0] - radius, p1[1] - radius, p1[2] + radius],
                [p1[0] + radius, p1[1] - radius, p1[2] + radius],
                [p1[0] + radius, p1[1] + radius, p1[2] + radius],
                [p1[0] - radius, p1[1] + radius, p1[2] + radius],
            ]
            box2 = [
                [p2[0] - radius, p2[1] - radius, p2[2] - radius],
                [p2[0] + radius, p2[1] - radius, p2[2] - radius],
                [p2[0] + radius, p2[1] + radius, p2[2] - radius],
                [p2[0] - radius, p2[1] + radius, p2[2] - radius],
                [p2[0] - radius, p2[1] - radius, p2[2] + radius],
                [p2[0] + radius, p2[1] - radius, p2[2] + radius],
                [p2[0] + radius, p2[1] + radius, p2[2] + radius],
                [p2[0] - radius, p2[1] + radius, p2[2] + radius],
            ]
            points.extend(box1 + box2)
            lines.extend([
                [i * 16, i * 16 + 1], [i * 16 + 1, i * 16 + 2], [i * 16 + 2, i * 16 + 3], [i * 16 + 3, i * 16],
                [i * 16 + 4, i * 16 + 5], [i * 16 + 5, i * 16 + 6], [i * 16 + 6, i * 16 + 7], [i * 16 + 7, i * 16 + 4],
                [i * 16, i * 16 + 4], [i * 16 + 1, i * 16 + 5], [i * 16 + 2, i * 16 + 6], [i * 16 + 3, i * 16 + 7],
                [i * 16 + 8, i * 16 + 9], [i * 16 + 9, i * 16 + 10], [i * 16 + 10, i * 16 + 11],
                [i * 16 + 11, i * 16 + 8],
                [i * 16 + 12, i * 16 + 13], [i * 16 + 13, i * 16 + 14], [i * 16 + 14, i * 16 + 15],
                [i * 16 + 15, i * 16 + 12],
                [i * 16 + 8, i * 16 + 12], [i * 16 + 9, i * 16 + 13], [i * 16 + 10, i * 16 + 14],
                [i * 16 + 11, i * 16 + 15],
            ])

        points = np.array(points)
        colors = [color for _ in range(len(lines))]

        wireframe = o3d.geometry.LineSet()
        wireframe.points = o3d.utility.Vector3dVector(points)
        wireframe.lines = o3d.utility.Vector2iVector(lines)
        wireframe.colors = o3d.utility.Vector3dVector(colors)

        return wireframe

    def place_valve_on_pipe(self, path_points, pipe_radius):
        valve_name = "Valve"
        self.ensure_object_list(valve_name)
        valve_attributes = self.scene_graph.graph.nodes[valve_name]
        valve_size = tuple(valve_attributes["Size"])
        color = self.color_map[valve_name]

        pipe_length = len(path_points)
        valve_position_index = random.randint(0, pipe_length - 1)
        valve_position = path_points[valve_position_index].copy()

        valve_position[0] = valve_position[0] - (pipe_radius / 2)
        valve_position[1] = valve_position[1] - (pipe_radius / 2)
        valve_position[2] -= (pipe_radius * 3)

        valve_position = tuple(map(int, valve_position))

        wireframe = self.visualizer.create_wireframe_box(valve_name, valve_size, color, translate=valve_position)
        self.placed_objects[valve_name].append((valve_position, valve_size, wireframe))
        self.helper.mark_grid(valve_position, valve_size)

    def bfs_traverse_and_place(self):
        queue = ["Floor", "Wall", "Ceiling"]
        visited = set(queue)

        self.place_floor()
        self.place_walls()
        self.place_ceiling()

        while queue:
            current_object = queue.pop(0)
            for neighbor in self.scene_graph.graph.successors(current_object):
                if neighbor not in visited:
                    if neighbor not in ["Valve", "Wire"]:
                        self.place_objects(neighbor)
                    queue.append(neighbor)
                    visited.add(neighbor)

        self.place_pipes_and_valve()


class PlacementHelper:
    def __init__(self, room_size, scene_graph):
        self.grid = np.zeros(room_size, dtype=bool)
        self.room_size = room_size
        self.scene_graph = scene_graph

    def mark_grid(self, position, size):
        x, y, z = map(int, position)
        sx, sy, sz = size
        self.grid[x:x + sx, y:y + sy, z:z + sz] = True

    def mark_grid_for_pipe(self, path_points, radius):
        for point in path_points:
            x, y, z = map(int, point)
            sx, sy, sz = int(radius * 2), int(radius * 2), int(radius * 2)
            self.grid[max(0, x - sx // 2):min(self.room_size[0], x + sx // 2 + 1),
            max(0, y - sy // 2):min(self.room_size[1], y + sy // 2 + 1),
            max(0, z - sz // 2):min(self.room_size[2], z + sz // 2 + 1)] = True

    def check_grounding(self, potential_points, object_name, size):
        object_attributes = self.scene_graph.graph.nodes[object_name]
        grounding = object_attributes.get('Conditions', {}).get('Grounding', 'Floor')
        distance = object_attributes.get('Conditions', {}).get('DistanceFromObjects', 0)
        wall_thickness = self.scene_graph.graph.nodes["Wall"]["Size"][0]

        grounded_points = []
        if grounding == 'Wall':
            if object_name == 'Door':
                y = 0
                for x in range(wall_thickness, self.room_size[0] - wall_thickness - size[0] + 1):
                    if not self.check_overlap((x, y, 0), size, distance):
                        grounded_points.append((x, y, 0))

            elif object_name == 'ElectricBox':
                x = self.room_size[0] - size[0]
                z = (self.room_size[2] // 2) - (size[2] // 2)
                for y in range(wall_thickness, self.room_size[1] - wall_thickness - size[1] + 1):
                    if not self.check_overlap((x, y, z), size, distance):
                        grounded_points.append((x, y, z))

            elif object_name == 'Valve':
                x = self.room_size[0] - size[0]
                z = (self.room_size[2] // 2) - (size[2] // 2)
                for y in range(wall_thickness, self.room_size[1] - wall_thickness - size[1] + 1):
                    if not self.check_overlap((x, y, z), size, distance):
                        grounded_points.append((x, y, z))

            elif object_name == 'Ladder':
                x = 0
                for y in range(wall_thickness, self.room_size[1] - wall_thickness - size[1] + 1):
                    if not self.check_overlap((x, y, 0), size, distance):
                        grounded_points.append((x, y, 0))

            elif object_name == 'Stairs':
                x_positions = [wall_thickness, self.room_size[0] - wall_thickness - size[0]]
                for x in x_positions:
                    if not self.check_overlap((x, 0, 0), size, distance):
                        grounded_points.append((x, 0, 0))

        elif grounding == 'Floor' or grounding == 'Ceiling':
            z_ground = 0 if grounding == 'Floor' else self.room_size[2] - size[2]
            for x, y, _ in potential_points:
                if not self.check_overlap((x, y, z_ground), size, distance):
                    grounded_points.append((x, y, z_ground))

        return grounded_points

    def check_overlap(self, position, size, distance):
        x, y, z = position
        sx, sy, sz = size
        expanded_x = max(0, x - distance)
        expanded_y = max(0, y - distance)
        expanded_sx = min(self.room_size[0], x + sx + distance)
        expanded_sy = min(self.room_size[1], y + sy + distance)

        if np.any(self.grid[expanded_x:expanded_sx, expanded_y:expanded_sy]):
            return True
        return False

    def check_pipe_overlap(self, path_points, radius):
        for point in path_points:
            x, y, z = map(int, point)
            sx, sy, sz = int(radius * 2), int(radius * 2), int(radius * 2)
            expanded_x = max(0, x - sx // 2)
            expanded_y = max(0, y - sy // 2)
            expanded_sx = min(self.room_size[0], x + sx // 2)
            expanded_sy = min(self.room_size[1], y + sy // 2)

            if np.any(self.grid[expanded_x:expanded_sx, expanded_y:expanded_sy]):
                return True
        return False

    def save_object(self, object_name, position, color):
        """Save the object as a point cloud (.txt) in the specified folder structure."""
        dataset_path = os.path.join(self.object_placer.dataset_name, f"Area_{self.object_placer.area_count}",
                                    f"room_{self.object_placer.room_count}", "Annotations")
        os.makedirs(dataset_path, exist_ok=True)

        # Track object count for naming convention
        if object_name not in self.object_placer.object_count:
            self.object_placer.object_count[object_name] = 0
        else:
            self.object_placer.object_count[object_name] += 1

        object_filename = f"{object_name.lower()}_{self.object_placer.object_count[object_name]}.txt"
        object_path = os.path.join(dataset_path, object_filename)

        # Simulating point cloud data for the object
        points = np.random.rand(100, 3) * 10  # Simulated (x, y, z)
        colors = np.random.rand(100, 3) * 255  # Simulated (r, g, b)
        point_cloud_data = np.hstack((points, colors))

        # Save the object as a point cloud .txt file
        np.savetxt(object_path, point_cloud_data, fmt='%.6f %.6f %.6f %.6f %.6f %.6f')


class Visualizer:
    def __init__(self, object_placer, dataset_name, area, room, point_noise, point_density):
        self.object_placer = object_placer
        self.visual_scene = []
        self.area = area
        self.room = room
        self.dataset_name = dataset_name
        self.object_counter = {}
        self.point_noise = point_noise / 100
        self.point_density = point_density

    def add_tube_to_scene(self, tube):
        pcd = sample_point_cloud_poisson(tube, self.point_density)
        self.save_object(pcd, 'pipe')
        self.visual_scene.append(pcd)

    def create_wireframe_box(self, object_name, size, color, translate=(0, 0, 0), grid_density=10):
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        points = [
            [0, 0, 0], [size[0], 0, 0], [size[0], size[1], 0], [0, size[1], 0],
            [0, 0, size[2]], [size[0], 0, size[2]], [size[0], size[1], size[2]], [0, size[1], size[2]]
        ]
        points = np.array(points) + np.array(translate)

        if object_name == 'Floor' or object_name == 'Ceiling':
            points = points.tolist()
            for i in range(1, grid_density):
                fraction = i / grid_density
                start = np.array([0, fraction * size[1], size[2]]) + np.array(translate)
                end = np.array([size[0], fraction * size[1], size[2]]) + np.array(translate)
                points.extend([start.tolist(), end.tolist()])
                lines.append([len(points) - 2, len(points) - 1])

                start = np.array([fraction * size[0], 0, size[2]]) + np.array(translate)
                end = np.array([fraction * size[0], size[1], size[2]]) + np.array(translate)
                points.extend([start.tolist(), end.tolist()])
                lines.append([len(points) - 2, len(points) - 1])

            colors = [color for _ in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
        else:
            colors = [color for _ in range(len(lines))]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

        # self.visual_scene.append(line_set)
        return line_set

    def load_and_place_mesh(self, pcd_path, size, translate, orientation, object_name):
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd_bounds = pcd.get_axis_aligned_bounding_box()
        current_extent = np.array(pcd_bounds.get_extent())

        if object_name == "Railing":
            pcd_bounds_min = pcd.get_min_bound()
            pcd_bounds_max = pcd.get_max_bound()

            # Calculate the extent of the point cloud in x, y, z dimensions
            current_extent = np.array(pcd_bounds_max) - np.array(pcd_bounds_min)

            # Calculate scaling factors for each axis
            scale_x = size[0] / current_extent[0]
            scale_y = size[1] / current_extent[1]
            scale_z = size[2] / current_extent[2]
            scale_factor_xy = min(scale_x, scale_y)
            scale_factors = np.array([scale_factor_xy, scale_factor_xy, scale_z])
            scale_transform = np.diag(np.append(scale_factors, [1]))
            pcd.transform(scale_transform)

            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(orientation))
            pcd.rotate(rotation_matrix, center=pcd.get_center())

            pcd_bounds = pcd.get_axis_aligned_bounding_box()
            pcd_center = np.array(pcd_bounds.get_center())
            desired_center = np.array(translate) + np.array(size) / 2
            translation_vector = desired_center - pcd_center

            pcd.translate(translation_vector)
            self.save_object(pcd, object_name)
            self.visual_scene.append(pcd)
            return pcd

        else:
            sorted_indices = np.argsort(-current_extent)
            sorted_size = np.array(size)[np.argsort(-np.array(size))]
            scale_factors = sorted_size / (current_extent[sorted_indices] + 1e-6)

            original_scale_factors = np.zeros_like(scale_factors)
            original_scale_factors[sorted_indices] = scale_factors
            scale_transform = np.diag(np.append(original_scale_factors, [1]))
            pcd.transform(scale_transform)

            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(orientation))
            pcd.rotate(rotation_matrix, center=pcd.get_center())

            pcd_bounds = pcd.get_axis_aligned_bounding_box()
            pcd_center = np.array(pcd_bounds.get_center())
            pcd_extent = np.array(pcd_bounds.get_extent())

            wireframe_center = np.array(translate) + np.array(size) / 2
            final_translation = wireframe_center - pcd_center

            if final_translation.ndim > 1:
                final_translation = final_translation[0]

            pcd.translate(final_translation)
            self.save_object(pcd, object_name)
            self.visual_scene.append(pcd)
            return pcd

    def load_meshes_for_objects(self, path, density):
        orientation_mapping = {
            'Desk': f'{path}/Orientations/desk.txt',
            'Column': f'{path}/Orientations/column.txt',
            'Chair': f'{path}/Orientations/chair.txt',
            'Stairs': f'{path}/Orientations/stairs.txt',
            'Door': f'{path}/Orientations/door.txt',
            'ElectricBox': f'{path}/Orientations/electricbox.txt',
            'Equipment': f'{path}/Orientations/equipment.txt',
            'Ladder': f'{path}/Orientations/ladder.txt',
            'Support': f'{path}/Orientations/support.txt',
            'Valve': f'{path}/Orientations/valve.txt',
            'Railing': f'{path}/Orientations/railing.txt',
        }

        orientation_indices = {}

        for key in orientation_mapping:
            with open(orientation_mapping[key], 'r') as file:
                lines = file.readlines()
            orientation_indices[key] = randrange(len(lines))

        mesh_mapping = {
            'Desk': f'{path}/PCD_Objects_{density}/desk/desk_{orientation_indices["Desk"] + 1}.pcd',
            'Column': f'{path}/PCD_Objects_{density}/column/column_{orientation_indices["Column"] + 1}.pcd',
            'Chair': f'{path}/PCD_Objects_{density}/chair/chair_{orientation_indices["Chair"] + 1}.pcd',
            'Stairs': f'{path}/PCD_Objects_{density}/stairs/stairs_{orientation_indices["Stairs"] + 1}.pcd',
            'Door': f'{path}/PCD_Objects_{density}/door/door_{orientation_indices["Door"] + 1}.pcd',
            'ElectricBox': f'{path}/PCD_Objects_{density}/electricbox/electricbox_{orientation_indices["ElectricBox"] + 1}.pcd',
            'Equipment': f'{path}/PCD_Objects_{density}/equipment/equipment_{orientation_indices["Equipment"] + 1}.pcd',
            'Ladder': f'{path}/PCD_Objects_{density}/ladder/ladder_{orientation_indices["Ladder"] + 1}.pcd',
            'Support': f'{path}/PCD_Objects_{density}/support/support_{orientation_indices["Support"] + 1}.pcd',
            'Valve': f'{path}/PCD_Objects_{density}/valve/valver_{orientation_indices["Valve"] + 1}.pcd',
            'Railing': f'{path}/PCD_Objects_{density}/railing/railing_{orientation_indices["Railing"] + 1}.pcd'
        }

        color_map = self.object_placer.color_map

        for object_name, data in self.object_placer.placed_objects.items():
            for index, values in enumerate(data):
                position, size, wireframe = values[:3]
                if object_name in mesh_mapping:
                    orientation_index = orientation_indices[object_name]
                    with open(f'{path}/Orientations/{object_name}.txt', 'r') as file:
                        lines = file.readlines()
                    orientation_value = lines[orientation_index].strip()
                    orientation = tuple(map(float, orientation_value.split(',')))
                    mesh_path = mesh_mapping[object_name]
                    mesh = self.load_and_place_mesh(mesh_path, size, position, orientation, object_name)
                    mesh.paint_uniform_color(color_map.get(object_name, [1.0, 1.0, 1.0]))
                    self.object_placer.placed_objects[object_name][index] = (position, size, (wireframe, mesh))

                elif object_name == "Pipe":
                    continue

    def visualize_3d_grid(self):
        o3d.visualization.draw_geometries(self.visual_scene)

    def visualize_graph(self, scene_graph):
        pos = nx.spring_layout(scene_graph.graph, k=2)
        plt.figure(figsize=(10, 8))
        nx.draw(scene_graph.graph, pos, with_labels=True, node_color='skyblue', node_size=4000, font_size=12)
        plt.title('Scene Graph Visualization')
        plt.show()

    def line_set_to_mesh(self, line_set, radius=0.02, resolution=60):
        mesh = o3d.geometry.TriangleMesh()
        points = np.asarray(line_set.points)
        lines = np.asarray(line_set.lines)
        colors = np.asarray(line_set.colors)

        for i, line in enumerate(lines):
            start = points[line[0]]
            end = points[line[1]]
            color = colors[i]
            cylinder = self.create_cylinder_between_points(start, end, radius, color, resolution)
            mesh += cylinder

        return mesh

    def create_cylinder_between_points(self, start, end, radius, color, resolution=60):
        height = np.linalg.norm(end - start)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution=resolution)
        cylinder.compute_vertex_normals()

        direction = (end - start) / height
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.dot(z_axis, direction))
        if np.linalg.norm(axis) > 0:
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            cylinder.rotate(rotation_matrix, center=(0, 0, 0))

        cylinder.translate(start)

        cylinder.paint_uniform_color(color)

        return cylinder

    def save_object(self, pcd, object_name):
        """Save the object point cloud as a .txt file in the specified dataset structure."""
        dataset_path = os.path.join(self.dataset_name, f"Area_{self.area}", f"room_{self.room}", "Annotations")
        os.makedirs(dataset_path, exist_ok=True)

        # Track object count for naming convention
        if object_name not in self.object_counter:
            self.object_counter[object_name] = 1
        else:
            self.object_counter[object_name] += 1

        object_filename = f"{object_name.lower()}_{self.object_counter[object_name]}.txt"
        object_path = os.path.join(dataset_path, object_filename)

        # Extract point cloud data (x, y, z, r, g, b) from the pcd
        points = np.asarray(pcd.points)  # (x, y, z)
        colors = np.asarray(pcd.colors) * 255  # (r, g, b) scaled to 0-255
        noise_array = np.random.normal(0, self.point_noise, points.shape)  # Create noise with mean 0 and stddev = noise
        points_with_noise = points + noise_array
        point_cloud_data = np.hstack((points_with_noise, colors))  # Combine (x, y, z) with (r, g, b)

        # Save the point cloud data to a .txt file without any extra headers
        np.savetxt(object_path, point_cloud_data, fmt='%.6f %.6f %.6f %.6f %.6f %.6f')


if __name__ == '__main__':


    number_of_areas = 10
    number_of_rooms = 45
    test_mode = True
    dataset_main = r'E:\PycharmProjects\Procedural-PointCloudForge\TestDataset'

    point_densities = [10000]
    point_noises = [5]

    for density in tqdm(point_densities, desc="Point Densities"):
        object_path = f'E:\PycharmProjects\Procedural-PointCloudForge'
        for noise in tqdm(point_noises, desc="Point Noises", leave=False):
            dataset_name = f"{dataset_main}\Datasets_{noise}_{density}_final"
            for area in tqdm(range(number_of_areas-1), desc="Areas", leave=False):
                for room in tqdm(range(number_of_rooms), desc="Rooms", leave=False):
                    scene_graph = SceneGraph()
                    scene_graph.parse_constraints('constraints.json')
                    object_placer = ObjectPlacer(scene_graph, (50, 50, 20), None, noise, density)
                    visualizer = Visualizer(object_placer, dataset_name, area + 1, room + 1, point_noise=noise,
                                            point_density=density)
                    object_placer.visualizer = visualizer

                    object_placer.generate_unique_colors()  # Generate unique colors for each scene
                    object_placer.bfs_traverse_and_place()
                    visualizer.load_meshes_for_objects(object_path, density)

                    if test_mode:
                        visualizer.visualize_3d_grid()
                        visualizer.visualize_graph(scene_graph)
                    # visualizer.save_scene("Datasets")
            create_alignment_files(dataset_name)
            made_datasets = dataset_name, dataset_name + '_lidar'
            copy_directory_with_files(made_datasets[0], made_datasets[1])
            process_rooms_in_areas(made_datasets[1])
            scale = 0.25
            for base_directory in made_datasets:
                merge_txt_files_in_annotations(base_directory)
                scale_dataset(base_directory, scale)
                create_alignment_files(base_directory)
                print(base_directory)
                rename_files_and_folders(base_directory, original_name='ladder', new_name='stairs')
                rename_files_and_folders(base_directory, original_name='desk', new_name='table')
                rename_files_and_folders(base_directory, original_name='pipe', new_name='chs')
                rename_files_and_folders(base_directory, original_name='support', new_name='beam')
                rename_files_and_folders(base_directory, original_name='valve', new_name='equipment')
                rename_files_and_folders(base_directory, original_name='electricbox', new_name='equipment')
                rename_files_and_folders(base_directory, original_name='railing', new_name='wall')
                rename_files_and_folders(base_directory, original_name='stairs', new_name='chs')
                #SpaceScene(base_directory, output_root_directory)
