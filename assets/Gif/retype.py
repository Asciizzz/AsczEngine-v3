import os

def rename_gif_to_png(folder_path):
    # Check if the folder path exists
    if not os.path.isdir(folder_path):
        print("The specified folder does not exist.")
        return

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has a .gif extension
        if filename.lower().endswith(".gif"):
            # Construct full file path
            old_file = os.path.join(folder_path, filename)
            # Create new filename with .png extension
            new_file = os.path.join(folder_path, filename.rsplit(".", 1)[0] + ".png")
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} to {new_file}")

# Replace 'your_folder_path_here' with the path to your folder
rename_gif_to_png('C:/Users/DPC/Downloads/VSCLMAO/ASCZENGINE/AsczEngine_v3/assets/Gif')
