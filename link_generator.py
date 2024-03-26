import requests
from bs4 import BeautifulSoup
import os
import re
from time import sleep


def download_images(base_folder, max_pages):
    with open("cat_breeds.txt") as file:            
        lines = file.readlines()
        for i, line in enumerate(lines):
            with open(f'\\links\\{line.replace("\n","")}_links.txt', 'w') as f: 
                search_term = line.strip()
                search_query = search_term.replace(' ', '+')

                for page in range(1, max_pages + 1):
                    url = f'https://images.search.yahoo.com/search/images?p={search_query}+cat&fr=yfp-t&ei=UTF-8&b={page * 28 + 1}'
                    #+cat&fr=yfp-t&ei=UTF-8&b={page * 28 + 1}

                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                    }
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                
                        img_urls = [img.get('href') for img in soup.find_all('a') if img.get('href') and img.get('href').startswith('/images/view')]
                        for link in img_urls:
                            f.writelines(f"{f.read}{link}\n")
                    else:
                        print('Bağlantı hatası:', response.status_code)

folder_path = 'C:\\Users\\ertug\\Desktop\\makine_ögrenmesi_proje'
download_images(folder_path,10)