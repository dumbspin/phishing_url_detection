__all__ = ["extract_domain_from_url", "setup_logging"]

from urllib.parse import urlparse
import logging
import sys

def setup_logging(logfile='pipeline.log'):
    """
    Set up logging to both a file and stdout. Call this once at the start of your program.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout)
        ]
    )

def extract_domain_from_url(url):
    """
    Safely extracts the domain (hostname) from a URL using urlparse and returns it in lowercase.
    If parsing fails, returns an empty string.
    """
    try:
        parsed = urlparse(url)
        if parsed.hostname:
            return parsed.hostname.lower()
        else:
            return ''
    except Exception:
        return '' 