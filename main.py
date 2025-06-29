from src.data_fetching import fetch_phishtank_data, fetch_alexa_top_sites, label_and_merge_datasets
from src.feature_extraction import extract_content_features, extract_url_features_bulk
from .utils import extract_domain_from_url, setup_logging
from src.train_model import train_url_classifier
import pandas as pd
import os

def main():
    setup_logging()
    # Step 1: Fetch data
    fetch_phishtank_data()
    fetch_alexa_top_sites()

    # Step 2: Extract content and URL features
    phish_df = pd.read_csv('data/phishtank_latest.csv')
    alexa_df = pd.read_csv('data/alexa_top_sites.csv')
    phish_urls = phish_df['url'].dropna().tolist()
    alexa_urls = alexa_df['url'].dropna().tolist()

    # Extract content features for phishing URLs (can also do for Alexa if desired)
    extract_content_features(phish_urls, csv_path='data/content_features.csv')
    # Extract URL features for both datasets and concatenate
    extract_url_features_bulk(phish_urls, csv_path='data/phish_url_features.csv')
    extract_url_features_bulk(alexa_urls, csv_path='data/alexa_url_features.csv')

    # Step 3: Merge and label datasets
    label_and_merge_datasets('data/phish_url_features.csv', 'data/alexa_url_features.csv', 'data/url_features.csv')

    # Step 4: Train the classifier
    train_url_classifier(features_csv='data/url_features.csv', model_path='models/url_xgb_model.joblib')

if __name__ == "__main__":
    main() 