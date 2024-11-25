
import os
import re
from collections import defaultdict


def rename_files_and_folders(root_dir, original_name, new_name):
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if we are inside an "Annotations" folder
        if dirpath.endswith("Annotations"):
            # Dictionary to keep track of the next available number for each new_name in the folder
            next_number_dict = {}

            # First pass: loop through all the files
            for filename in filenames:
                # Extract the part before the first underscore (or use the whole name if no underscores)
                object_name = filename.split('_')[0]

                # If the object matches the original_name or is already the new_name
                if object_name == original_name or object_name == new_name:
                    # Initialize the next number for the new_name if not already set
                    if new_name not in next_number_dict:
                        next_number_dict[new_name] = 0

                    # Increment the next available number for the new_name
                    next_number_dict[new_name] += 1
                    new_number = next_number_dict[new_name]

                    # Create the new filename with the next available number
                    new_filename = f"{new_name}_{new_number}.txt"
                    new_file = os.path.join(dirpath, new_filename)

                    # Handle file conflicts: If the new filename already exists, find the next available number
                    while os.path.exists(new_file):
                        next_number_dict[new_name] += 1
                        new_number = next_number_dict[new_name]
                        new_filename = f"{new_name}_{new_number}.txt"
                        new_file = os.path.join(dirpath, new_filename)

                    # Construct full paths for the old and new filenames
                    old_file = os.path.join(dirpath, filename)

                    # Rename the file
                    os.rename(old_file, new_file)
                    print(f"Renamed file: {old_file} to {new_file}")
            rename_files_consecutively(dirpath)
    print("Renaming completed for all Annotations folders.")

def rename_files_consecutively(dirpath):
    # Dictionary to store file lists by their prefix
    file_dict = defaultdict(list)

    # Regex pattern to match files with the prefix and number (e.g., chair_1.txt, clutter_5.txt)
    pattern = re.compile(r'([a-zA-Z]+)_(\d+)\.txt')

    # Go through all files in the directory
    for f in os.listdir(dirpath):
        match = pattern.match(f)
        if match:
            prefix = match.group(1)  # Capture the prefix (e.g., "chair", "clutter")
            number = int(match.group(2))  # Capture the number as an integer
            file_dict[prefix].append((f, number))

    # Process each prefix group
    for prefix, files in file_dict.items():
        # Sort files by their numeric part
        files.sort(key=lambda x: x[1])

        # Rename files to make the numbering consecutive
        for index, (original_file, _) in enumerate(files, start=1):
            new_file = f"{prefix}_{index}.txt"
            original_path = os.path.join(dirpath, original_file)
            new_path = os.path.join(dirpath, new_file)

            if original_file != new_file:
                print(f"Renaming {original_file} to {new_file}")
                os.rename(original_path, new_path)