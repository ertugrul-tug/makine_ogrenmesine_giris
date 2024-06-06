import scripts.duplicate_image_remover as duplicate_image_remover
import scripts.web_image_scrapper as web_image_scrapper

base_folder = 'C:\\Users\\ertug\\Desktop\\Folder\\makine_ögrenmesi_proje' # declaring the folder path

with open("cat_breeds.txt") as file: 
    cat_breeds = file.readlines() # getting names of cat breed types

def main():
    
    max_pages = 14

    # Download images
    #web_image_scrapper.download_images(base_folder, max_pages)

    # Remove duplicates
    duplicate_image_remover.remove_duplicates(base_folder, cat_breeds)

if __name__ == "__main__":
    main()
