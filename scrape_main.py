from scrape_utils import scrape_suburls,download_images_and_docs
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='Scrape suburls')
    parser.add_argument('--url', type=str, help='URL to scrape')
    args = parser.parse_args()

    url = args.url
    results = scrape_suburls(url)
    base_url = url.split('/')[2].split('.')[1]
    os.makedirs(base_url, exist_ok=True)
    with open(f'{base_url}/results.json', 'w') as f:
        json.dump(results, f)

    download_images_and_docs(results, base_url)

if __name__ == '__main__':
    main()