import os
import hashlib

hashes = set() # creating empty object to store hashes

def remove_duplicates(base_folder, folder_names):
    base_folder = base_folder + '\\images'
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