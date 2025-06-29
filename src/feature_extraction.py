__all__ = [
    "extract_url_features",
    "extract_url_features_bulk",
    "fetch_whois_for_domains",
    "get_whois_features"
]

import os
import pandas as pd
import re
from urllib.parse import urlparse
import whois
from datetime import datetime
import time
import logging
import sys
from .utils import setup_logging

# If you need to import from utils or other src modules, use relative imports
# from .utils import extract_domain_from_url  # Uncomment if needed

# Set up logging to file and stdout (if not already set)
setup_logging()

def extract_url_features(url, brand_keywords=None, suspicious_tlds=None):
    """
    Extracts URL-based features for phishing detection.
    Features:
        - url_length
        - num_dots
        - uses_ip_address
        - has_at
        - has_dash
        - has_special_chars
        - suspicious_tld
        - has_brand_keyword
    """
    if brand_keywords is None:
        brand_keywords = ['google', 'paypal', 'apple', 'amazon', 'microsoft', 'facebook', 'bank']
    if suspicious_tlds is None:
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work', '.support', '.info']

    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    url_length = len(url)
    num_dots = hostname.count('.')
    # Check for IP address in hostname
    uses_ip_address = bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname))
    has_at = '@' in url
    has_dash = '-' in hostname
    has_special_chars = any(c in url for c in ['$', '%', '&', '=', '?', '#', '_', '~'])
    suspicious_tld = any(hostname.endswith(tld) for tld in suspicious_tlds)
    has_brand_keyword = any(kw in url.lower() for kw in brand_keywords)
    return {
        'url': url,
        'url_length': url_length,
        'num_dots': num_dots,
        'uses_ip_address': uses_ip_address,
        'has_at': has_at,
        'has_dash': has_dash,
        'has_special_chars': has_special_chars,
        'suspicious_tld': suspicious_tld,
        'has_brand_keyword': has_brand_keyword
    }

def extract_url_features_bulk(urls, csv_path='data/url_features.csv'):
    """
    Extracts URL-based features for a list of URLs and saves them to a CSV file.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    features = [extract_url_features(url) for url in urls]
    df = pd.DataFrame(features)
    df.to_csv(csv_path, index=False)
    print(f"Saved URL features for {len(df)} URLs to {csv_path}")

def get_whois_features(domain, cache_path='data/whois_cache.csv'):
    """
    Takes a domain name and returns WHOIS features: domain age (in days), registrar, creation date, expiration date.
    Uses a CSV cache to avoid redundant WHOIS lookups.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # Load cache if exists
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
        if domain in cache_df['domain'].values:
            row = cache_df[cache_df['domain'] == domain].iloc[0]
            return {
                'domain': row['domain'],
                'domain_age_days': row['domain_age_days'],
                'registrar': row['registrar'],
                'creation_date': row['creation_date'],
                'expiration_date': row['expiration_date']
            }
    # If not cached, fetch WHOIS info
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        expiration_date = w.expiration_date
        registrar = w.registrar if hasattr(w, 'registrar') else None
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        if creation_date and isinstance(creation_date, datetime):
            domain_age = (datetime.utcnow() - creation_date).days
        else:
            domain_age = -1
        result = {
            'domain': domain,
            'domain_age_days': domain_age,
            'registrar': registrar,
            'creation_date': creation_date,
            'expiration_date': expiration_date
        }
    except Exception as e:
        logging.warning(f"[get_whois_features] Failed to fetch WHOIS for {domain}: {type(e).__name__}: {e}")
        result = {
            'domain': domain,
            'domain_age_days': -1,
            'registrar': None,
            'creation_date': None,
            'expiration_date': None
        }
    # Append to cache
    new_row = pd.DataFrame([result])
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
        # Remove duplicates, keep first
        cache_df = cache_df.drop_duplicates(subset=['domain'], keep='first')
        cache_df.to_csv(cache_path, index=False)
    else:
        new_row.to_csv(cache_path, index=False)
    return result

def fetch_whois_for_domains(domains, csv_path='data/whois_features.csv'):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results = []
    for domain in domains:
        features = get_whois_features(domain)
        results.append(features)
        time.sleep(1)  # Sleep to avoid rate-limiting
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved WHOIS features for {len(df)} domains to {csv_path}")

# If running as __main__, use absolute imports from src.<module> (already using relative imports for module use) 