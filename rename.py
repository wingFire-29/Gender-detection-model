import os

def batch_rename(folder_path, prefix):
    """Renames all files in the given folder with a sequential naming pattern."""
    files = os.listdir(folder_path)  # List all files in the folder
    files = sorted(files)  # Sort files to keep order (optional)

    for i, filename in enumerate(files):
        # Extract the file extension
        file_extension = os.path.splitext(filename)[1]

        # Create a new filename, e.g., male1.jpg, male2.png, etc.
        new_name = f"{prefix}{i + 1}{file_extension}"

        # Construct full file paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        # Check if the new path already exists
        if os.path.exists(new_path):
            print(f"Conflict: {new_path} already exists. Skipping {old_path}.")
            continue  # Skip renaming if a conflict occurs

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

# Specify your folder path and prefix
folder_path = r"D:\college\Projects\Tech-She\age and gender detection\dataset100\female"
prefix = "female"

# Call the function
batch_rename(folder_path, prefix)
import os

def batch_rename(folder_path, prefix):
    """Renames all files in the given folder with a sequential naming pattern."""
    files = os.listdir(folder_path)  # List all files in the folder
    files = sorted(files)  # Sort files to keep order (optional)

    for i, filename in enumerate(files):
        # Extract the file extension
        file_extension = os.path.splitext(filename)[1]

        # Create a new filename, e.g., male1.jpg, male2.png, etc.
        new_name = f"{prefix}{i + 1}{file_extension}"

        # Construct full file paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        # Check if the new path already exists
        if os.path.exists(new_path):
            print(f"Conflict: {new_path} already exists. Skipping {old_path}.")
            continue  # Skip renaming if a conflict occurs

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

# Specify your folder path and prefix
folder_path = r"D:\college\Projects\Tech-She\age and gender detection\testset"
prefix = ""

# Call the function
batch_rename(folder_path, prefix)
