import requests
from bs4 import BeautifulSoup
import os

folder_path = 'C:\\Users\\ertug\\Desktop\\makine_ögrenmesi_proje' # declaring the folder path as usual

def download_images(base_folder, max_pages):
    with open("cat_breeds.txt") as file:
        lines = file.readlines() # getting names of cat breed types
        
        for i, line in enumerate(lines):
            search_term = line.strip() # getting breed name in each loop
            search_query = search_term.replace(' ', '+') # setting breed name in order to use it in https request
            print("indirilen kategori: {}\n[".format(search_term), end = "") # printing the breed name and starting the loading bar

            save_folder = os.path.join(f'{base_folder}', search_term) # using os to assign the save path to versitality across other platforms
            os.makedirs(save_folder, exist_ok=True) # again same deal, making directory if it doesn't exist with versitality in mind

            for page in range(1, max_pages + 1):
                url = f'https://images.search.yahoo.com/search/images?p={search_query}+cat&fr=yfp-t&ei=UTF-8&b={page * 28 + 1}' # changing the url according to our looped breed type and also page count

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3' # assigning user agent to be able to make requests
                }
                response = requests.get(url, headers=headers) # requesting the html file and storing it in a variable

                if response.status_code == 200: # The HTTP 200 OK success status response code indicates that the request has succeeded. A 200 response is cacheable by default.
                    soup = BeautifulSoup(response.text, 'html.parser') # Giving soup the html variable as text so it can parse
                
                    img_tags = soup.find_all('img') # extracting all 'img' type divisions
                    img_urls = [img.get('src') for img in img_tags if img.get('src') and img.get('src').startswith('https')] # every 'img' division has src value assigned that holds the thumbnail image's url

                    for i, img_url in enumerate(img_urls): # try for every number of thumbnail's urls if not print the error code (out of index etc.) 
                        try:
                            img_data = requests.get(img_url, headers=headers).content # requesting the image from the 'src' value
                            with open(os.path.join(save_folder, f'image_{(page - 1) * 100 + i}.jpg'), 'wb') as f: # opening the '.jpg' file as writable
                                f.write(img_data) # writing the image data into that '.jpg' file
                        except Exception as e:
                            print(f'Hata: {e}') # Handling and printing if any error occurs
                else:
                    print('Bağlantı hatası:', response.status_code) # if status code is different than 200 (500, 404, etc.) printing the status code to debug
                print("=", end = "") # updating the loading bar every page loop
            print("] indirme tamamlandı") # after each breed type finishes downloading close the loading bar

download_images(folder_path,10) # call our function and giving it path and number of pages to loop