import requests
import csv
import zipfile
import io
import pandas as pd
from datetime import datetime
import os
import whois
import time
from .feature_extraction import extract_url_features, extract_url_features_bulk, fetch_whois_for_domains, get_whois_features
from .utils import extract_domain_from_url
from .content_feature_extraction import extract_content_features

def fetch_phishtank_data(csv_path='data/phishtank_latest.csv'):
    """
    Fetch the latest phishing URLs from PhishTank and save to a CSV file.
    Columns: url, submission_time, verified
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # PhishTank public data URL (CSV)
    phishtank_url = 'http://data.phishtank.com/data/online-valid.csv'
    response = requests.get(phishtank_url)
    response.raise_for_status()
    decoded_content = response.content.decode('utf-8')
    reader = csv.DictReader(decoded_content.splitlines())
    rows = []
    for row in reader:
        rows.append({
            'url': row['url'],
            'submission_time': row['submission_time'],
            'verified': row['verified']
        })
    df = pd.DataFrame(rows)
    # Strip whitespace from url
    df['url'] = df['url'].str.strip()
    # Convert submission_time to datetime
    df['submission_time'] = pd.to_datetime(df['submission_time'], errors='coerce')
    # Remove duplicate URLs
    df = df.drop_duplicates(subset=['url'])
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} phishing URLs to {csv_path}")

def fetch_alexa_top_sites(csv_path='data/alexa_top_sites.csv', n=1000):
    """
    Download Alexa (or similar) top sites and save to a CSV file.
    Columns: url
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Tranco is a modern alternative to Alexa, as Alexa is discontinued
    tranco_url = f'https://tranco-list.eu/top-{n}.csv'
    response = requests.get(tranco_url)
    response.raise_for_status()
    decoded_content = response.content.decode('utf-8')
    reader = csv.reader(decoded_content.splitlines())
    urls = []
    for i, row in enumerate(reader):
        if i == 0:
            continue  # skip header
        if len(row) > 1 and row[1].strip():
            domain = row[1].strip().lower()
            full_url = f'https://{domain}'
            urls.append({'url': full_url})
    df = pd.DataFrame(urls)
    # Remove duplicates
    df = df.drop_duplicates(subset=['url'])
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} top sites to {csv_path}")

def label_and_merge_datasets(phish_csv, alexa_csv, output_csv):
    """
    Reads the PhishTank CSV and assigns a label of 1.
    Reads the Alexa/Tranco CSV and assigns a label of 0.
    Concatenates both into a single DataFrame and saves it to output_csv.
    """
    phish_df = pd.read_csv(phish_csv)
    phish_df['label'] = 1
    alexa_df = pd.read_csv(alexa_csv)
    alexa_df['label'] = 0
    merged_df = pd.concat([phish_df, alexa_df], ignore_index=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved merged and labeled dataset to {output_csv}")

if __name__ == "__main__":
    fetch_phishtank_data()
    # Load URLs from phishing CSV and extract content features
    phishing_df = pd.read_csv('data/phishtank_latest.csv')
    urls = phishing_df['url'].dropna().tolist()
    extract_content_features(urls)

    # Preprocess URLs to extract unique, valid domains for WHOIS
    domains = [extract_domain_from_url(url) for url in urls]
    domains = [d for d in set(domains) if d]  # unique and non-empty
    fetch_whois_for_domains(domains)