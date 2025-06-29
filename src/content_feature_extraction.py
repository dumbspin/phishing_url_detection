import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import logging
import sys
from .utils import extract_domain_from_url, setup_logging  # if needed
from .feature_extraction import extract_url_features  # if needed

# Set up logging to file and stdout
setup_logging()

def get_visible_text(soup):
    # Remove script, style, and invisible elements
    for tag in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_content_features(urls, csv_path='data/content_features.csv', failed_path='data/failed_urls.csv'):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results = []
    failed_urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    }
    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            title = soup.title.string.strip() if soup.title and soup.title.string else ''
            # Extract all meta tag content
            meta_tags = soup.find_all('meta')
            meta_content = []
            for tag in meta_tags:
                if tag.get('content'):
                    meta_content.append(tag.get('content').strip())
            meta = ' | '.join(meta_content)
            visible_text = get_visible_text(soup)
        except Exception as e:
            logging.warning(f"[extract_content_features] Failed to extract content from {url}: {type(e).__name__}: {e}")
            title = ''
            meta = ''
            visible_text = ''
            failed_urls.append({'url': url, 'error': str(e)})
        results.append({
            'url': url,
            'title': title,
            'meta': meta,
            'body_text': visible_text
        })
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved content features for {len(df)} URLs to {csv_path}")
    if failed_urls:
        failed_df = pd.DataFrame(failed_urls)
        failed_df.to_csv(failed_path, index=False)
        logging.warning(f"Saved {len(failed_df)} failed URLs to {failed_path}")

if __name__ == "__main__":
    # Example: load URLs from a CSV file (e.g., data/phishtank_latest.csv or data/alexa_top_sites.csv)
    input_csv = 'data/phishtank_latest.csv'  # Change as needed
    url_col = 'url'
    urls = pd.read_csv(input_csv)[url_col].dropna().tolist()
    extract_content_features(urls) 