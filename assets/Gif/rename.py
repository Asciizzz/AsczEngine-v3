import os

def rename_files_to_frame_format(folder_path):
    # Check if the folder path exists
    if not os.path.isdir(folder_path):
        print("The specified folder does not exist.")
        return

    # Get a sorted list of all .png files in the folder
    png_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".png")])

    # Rename each .png file
    for index, filename in enumerate(png_files, start=0):
        # Construct full old file path
        old_file = os.path.join(folder_path, filename)
        # Create new filename with the 3-digit frame format
        new_filename = f"frame_{index:03}.png"
        new_file = os.path.join(folder_path, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} to {new_file}")

# Replace 'your_folder_path_here' with the path to your folder
rename_files_to_frame_format('C:/Users/DPC/Downloads/VSCLMAO/ASCZENGINE/AsczEngine_v3/assets/Gif')
