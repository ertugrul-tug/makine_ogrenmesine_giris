import os
import hashlib

hashes = set() # creating empty object to store hashes
folder_path = 'C:\\Users\\ertug\\Desktop\\makine_ögrenmesi_proje\\images' # declaring the folder path as usual

with open("cat_breeds.txt") as file: 
    cat_breeds = file.readlines() # getting names of cat breed types

def remove_duplicates(base_folder, folder_names):
    for i, cat_breed in enumerate(folder_names):
        folder = f'{cat_breed.replace("\n","")}' # getting breed name in each loop and getting rid of the break line "\n" from stirng value
        directory = (base_folder + '\\' + folder) # setting the directory to get same hashes from in order to delete same images
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename) # setting path of the image
            digest = hashlib.sha1(open(path,'rb').read()).digest() # getting hash value of image and converting it to store and compare
            if digest not in hashes:
                hashes.add(digest) # if hash value isn't stored, add to the object created earlyer
            else:
                os.remove(path) # if the hash value already stored simply delete

remove_duplicates(folder_path, cat_breeds) # call our function and give it path and names
