from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
import re
import os
import json


def process_subpage_urls(base_url, subpage_urls):
    url_dict = {}
    subpage_urls = sorted(list(set(subpage_urls)))
    base_domain = urlparse(base_url).netloc
    for url in subpage_urls:
        if url.startswith("/"):
            final_url = urljoin(base_url, url)
            url_format = url.replace("/", "_")[1:]
            url_dict[url_format] = final_url
        else:
            parsed_url = urlparse(url)
            if parsed_url.netloc == base_domain:
                url_format = url.replace("/", "_")
                url_dict[url_format] = url
    return url_dict


def scraperapi(url):
    payload = {'api_key': '18bdd66b4323c03e8983e588438b1012', 'url': url}
    response = requests.get('http://api.scraperapi.com', params=payload)
    return response

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace("\n", " ").replace("\t", " ").replace("\xa0", " ")
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def extract_images_from_js(html_content):
    js_arrays = re.findall(r'\[(?:[^\]]*?(?:\'|\")(?:https?:)?\/\/[^\'\"]*\.(?:png|jpg|jpeg|gif|svg)(?:\'|\")[^\]]*?)\]', html_content)
    
    image_urls = []
    for arr in js_arrays:
        try:
            urls = json.loads(arr)
            image_urls.extend([url for url in urls if isinstance(url, str) and url.startswith(('http://', 'https://', '//'))])
        except json.JSONDecodeError:
            pass
    
    # Look for individual image URLs
    individual_urls = re.findall(r'(?:\'|\")(?:https?:)?\/\/[^\'\"]*\.(?:png|jpg|jpeg|gif|svg)(?:\'|\")', html_content)
    image_urls.extend([url.strip('\'"') for url in individual_urls])
    
    return list(set(image_urls))

def process_suburl(url):
    response = scraperapi(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract text
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
    visible_text = soup.get_text(separator=' ', strip=True)
    visible_text = clean_text(visible_text)
    
    # Extract image URLs and convert to absolute URLs
    images = []
    for img in soup.find_all('img'):
        if 'src' in img.attrs:
            img_url = img['src']
            if not img_url.startswith('data:'):  # Filter out data URIs
                if not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(url, img_url)
                images.append(img_url)

    # Extract image URLs from JavaScript
    images.extend(extract_images_from_js(response.text))
    
    # Extract document URLs and convert to absolute URLs
    docs = []
    for a in soup.find_all('a'):
        if 'href' in a.attrs and a['href'].lower().endswith(('.pdf', '.doc', '.docx')):
            doc_url = a['href']
            if not doc_url.startswith(('http://', 'https://')):
                doc_url = urljoin(url, doc_url)
            docs.append(doc_url)
    
    return {
        "text": visible_text,
        "images": images,
        "docs": docs
    }

def scrape_suburls(url):
    response = scraperapi(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    subpage_urls = []
    links = soup.find_all("a")
    for link in links:
        if 'href' in link.attrs:
            subpage_urls.append(link["href"])
    
    url_dict = process_subpage_urls(url, subpage_urls)
    
    result_dict = {}
    
    with ThreadPoolExecutor(max_workers=18) as executor:
        future_to_url = {executor.submit(process_suburl, url): key for key, url in url_dict.items()}
        
        for future in tqdm(as_completed(future_to_url), total=len(future_to_url)):
            key = future_to_url[future]
            try:
                result_dict[key] = future.result()
            except Exception as exc:
                print(f'{key} generated an exception: {exc}')
    
    return result_dict

def download_and_save_image(image_urls, base_url):
    save_dir = f'{base_url}/images'
    os.makedirs(save_dir, exist_ok=True)

    for idx, image_url in enumerate(image_urls):
        image = requests.get(image_url)
        with open(f'{save_dir}/{idx}.jpg', 'wb') as f:
            f.write(image.content)

def download_and_save_doc(doc_urls, base_url):
    save_dir = f'{base_url}/docs'
    os.makedirs(save_dir, exist_ok=True)

    for idx, doc_url in enumerate(doc_urls):
        doc = requests.get(doc_url)
        with open(f'{save_dir}/{idx}.pdf', 'wb') as f:
            f.write(doc.content)

def download_images_and_docs(results, base_url):
    for key, value in results.items():
        download_and_save_image(value["images"], base_url)
        download_and_save_doc(value["docs"], base_url)