#!/usr/bin/env python3
"""
extract_10k_content.py

This script extracts the actual 10K content from SEC EDGAR filings
by parsing the HTML content and extracting readable text.

Dependencies:
    pip install requests beautifulsoup4

Usage:
    python extract_10k_content.py
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re

# Required headers for SEC EDGAR requests
HEADERS = {
    "User-Agent": "Sasha sashalodh@gmail.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}

OUTPUT_DIR = "extracted_10k_content"

# URLs from the scraper output
FILING_URLS = {
    "Apple_2024": "https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
    "Apple_2023": "https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
    "Microsoft_2025": "https://www.sec.gov/ix?doc=/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm",
    "Microsoft_2024": "https://www.sec.gov/ix?doc=/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
    "Tesla_2025": "https://www.sec.gov/ix?doc=/Archives/edgar/data/1318605/000162828025003063/tsla-20241231.htm",
    "Tesla_2024": "https://www.sec.gov/ix?doc=/Archives/edgar/data/1318605/000162828024002390/tsla-20231231.htm"
}

def clean_text(text):
    """Clean and format extracted text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\$\%]', '', text)
    return text.strip()

def extract_10k_content(url, company_name):
    """
    Extract 10K content from a filing URL.
    """
    try:
        print(f"Extracting content from {company_name}...")
        print(f"URL: {url}")
        
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text = soup.get_text()
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Save to file
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"{company_name}_10K_content.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        print(f"Saved {len(cleaned_text)} characters to {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error extracting content for {company_name}: {e}")
        return False

def main():
    """
    Extract content from all 10K filings.
    """
    print("Extracting 10K content from SEC EDGAR filings...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    success_count = 0
    
    for company_filing, url in FILING_URLS.items():
        if extract_10k_content(url, company_filing):
            success_count += 1
        
        # Rate limiting - be respectful to SEC servers
        time.sleep(1)
    
    print(f"\nCompleted! Successfully extracted {success_count}/{len(FILING_URLS)} 10K filings")

if __name__ == "__main__":
    main()
