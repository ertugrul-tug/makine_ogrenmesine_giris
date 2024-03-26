import requests
from bs4 import BeautifulSoup
import os
import re


def download_images(base_folder, max_pages):
    with open("cat_breeds.txt") as file:            
        lines = file.readlines()
        for i, line in enumerate(lines):
            with open(f'\\links\\{line.replace("\n","")}_links.txt', 'r') as f: 
                searchlines = f.readlines()
                for s, searchline in enumerate(searchlines):
                    search_term = searchline[72:]
                    print(search_term)
                    url = f'https://images.search.yahoo.com+{search_term}'
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                    }
                    response = requests.get(url, headers=headers)
                    try:
                        if response.status_code == 200:
                            lastsoup = BeautifulSoup(lastresponse.text, 'html.parser')
                            img_url = [img.get('src') for img in soup.find_all('img') if img.get('src') and img.get('src').startswith('https')]
                            img_data = requests.get(img_url, headers=headers).content
                            print(img_url)
                            with open(os.path.join(save_folder, f'image_{(page - 1) * 100 + i}.jpg'), 'wb') as f:
                                f.write(img_data)
                    except Exception as e:
                        print(f'Hata: {e}')

folder_path = 'C:\\Users\\ertug\\Desktop\\makine_ögrenmesi_proje'
download_images(folder_path,10)