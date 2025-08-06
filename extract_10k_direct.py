#!/usr/bin/env python3
"""
extract_10k_direct.py

This script extracts the actual 10K content by accessing the direct HTML files
instead of the XBRL viewer pages.

Dependencies:
    pip install requests beautifulsoup4

Usage:
    python extract_10k_direct.py
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

# Direct URLs to the actual HTML files (removing the XBRL viewer wrapper)
FILING_URLS = {
    "Apple_2024": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
    "Apple_2023": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
    "Microsoft_2025": "https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm",
    "Microsoft_2024": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
    "Tesla_2025": "https://www.sec.gov/Archives/edgar/data/1318605/000162828025003063/tsla-20241231.htm",
    "Tesla_2024": "https://www.sec.gov/Archives/edgar/data/1318605/000162828024002390/tsla-20231231.htm"
}

def clean_text(text):
    """Clean and format extracted text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Keep structure by preserving some line breaks for sections
    text = re.sub(r'\.([A-Z])', r'.\n\n\1', text)  # New paragraph after sentences starting with capitals
    return text.strip()

def extract_section_content(soup, section_name):
    """Extract specific sections of the 10K."""
    content = []
    
    # Look for section headers
    section_patterns = [
        rf"Item\s+1\s*[\.\-]*\s*{re.escape(section_name)}",
        rf"ITEM\s+1\s*[\.\-]*\s*{re.escape(section_name.upper())}",
        rf"{re.escape(section_name)}"
    ]
    
    for pattern in section_patterns:
        headers = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
        for header in headers:
            parent = header.parent
            if parent:
                # Try to get content after this header
                content.append(parent.get_text())
    
    return '\n\n'.join(content)

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
        
        # Try to extract key sections
        sections = {
            "Business": extract_section_content(soup, "Business"),
            "Risk Factors": extract_section_content(soup, "Risk Factors"),
            "Financial Data": extract_section_content(soup, "Selected Financial Data"),
            "MD&A": extract_section_content(soup, "Management's Discussion"),
            "Financial Statements": extract_section_content(soup, "Financial Statements")
        }
        
        # If sections are empty, get all text
        if all(not section.strip() for section in sections.values()):
            text = soup.get_text()
        else:
            text = "\n\n=== BUSINESS ===\n\n" + sections["Business"] + \
                   "\n\n=== RISK FACTORS ===\n\n" + sections["Risk Factors"] + \
                   "\n\n=== FINANCIAL DATA ===\n\n" + sections["Financial Data"] + \
                   "\n\n=== MD&A ===\n\n" + sections["MD&A"] + \
                   "\n\n=== FINANCIAL STATEMENTS ===\n\n" + sections["Financial Statements"]
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Save to file
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"{company_name}_10K_full_content.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        print(f"Saved {len(cleaned_text)} characters to {filename}")
        
        # Also save first 2000 characters as a preview
        preview_filename = f"{company_name}_10K_preview.txt"
        preview_filepath = os.path.join(OUTPUT_DIR, preview_filename)
        with open(preview_filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_text[:2000] + "\n\n[Content continues...]")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error extracting content for {company_name}: {e}")
        return False

def main():
    """
    Extract content from all 10K filings.
    """
    print("Extracting 10K content from SEC EDGAR filings (direct HTML)...")
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
