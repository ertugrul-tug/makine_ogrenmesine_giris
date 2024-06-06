from os import listdir
from os.path import join, isdir
from PIL import Image

base_folder = r'C:\Users\ertug\Desktop\Folder\makine_ögrenmesi_proje\images'
bad_files = []

# Iterate through each breed folder
for breed_folder in listdir(base_folder):
    breed_folder_path = join(base_folder, breed_folder)
    if isdir(breed_folder_path):
        # Iterate through each image file in the breed folder
        for filename in listdir(breed_folder_path):
            if filename.endswith('.jpg'):
                file_path = join(breed_folder_path, filename)
                try:
                    img = Image.open(file_path)  # Open the image file
                    img.verify()  # Verify that it is, in fact, an image
                except (IOError, SyntaxError) as e:
                    print('Bad file:', file_path)
                    bad_files.append(file_path)

print(f"Total bad files: {len(bad_files)}")
if bad_files:
    print("List of bad files:")
    for bad_file in bad_files:
        print(bad_file)
