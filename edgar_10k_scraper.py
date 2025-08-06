#!/usr/bin/env python3
"""
edgar_10k_scraper.py

This script searches for the most recent 10-K filings and downloads the 5 most recent ones to a local folder.

Dependencies:
    pip install requests beautifulsoup4

Usage:
    python edgar_10k_scraper.py
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import json
from urllib.parse import urljoin

# SEC EDGAR base URLs
EDGAR_BASE = "https://www.sec.gov"
FILINGS_BASE = "https://data.sec.gov/submissions"

# Required headers for SEC EDGAR requests
HEADERS = {
    "User-Agent": "Sasha sashalodh@gmail.com",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

OUTPUT_DIR = "edgar_10k"


def get_company_filings(cik):
    """
    Get filings for a specific company using CIK.
    """
    # Format CIK to 10 digits with leading zeros
    cik_formatted = str(cik).zfill(10)
    url = f"{FILINGS_BASE}/CIK{cik_formatted}.json"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching company filings for CIK {cik}: {e}")
        return None


def find_recent_10k_filings(filings_data, limit=5):
    """
    Extract recent 10-K filings from company submissions data.
    """
    recent_filings = filings_data.get("filings", {}).get("recent", {})
    forms = recent_filings.get("form", [])
    dates = recent_filings.get("filingDate", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    
    ten_k_filings = []
    for i, form in enumerate(forms):
        if form == "10-K" and len(ten_k_filings) < limit:
            ten_k_filings.append({
                "form": form,
                "filingDate": dates[i],
                "accessionNumber": accession_numbers[i],
                "cik": filings_data.get("cik")
            })
    
    return ten_k_filings


def construct_filing_url(cik, accession_number):
    """
    Construct the URL to the filing detail page.
    """
    cik_formatted = str(cik).zfill(10)
    accession_clean = accession_number.replace("-", "")
    return f"{EDGAR_BASE}/Archives/edgar/data/{cik}/{accession_clean}/{accession_number}-index.html"


def get_10k_document_url(filing_detail_url):
    """
    Parse the filing detail page to find the primary 10-K document.
    """
    try:
        response = requests.get(filing_detail_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for the document table
        table = soup.find("table", {"summary": "Document Format Files"})
        if not table:
            print("Document table not found")
            return None
        
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 4:
                # Check if this row contains the 10-K document
                doc_type = cells[3].get_text(strip=True)
                if "10-K" in doc_type:
                    link_cell = cells[2]
                    link = link_cell.find("a")
                    if link and link.get("href"):
                        return urljoin(EDGAR_BASE, link["href"])
        
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error parsing filing detail page: {e}")
        return None


def download_document(doc_url, filename, output_dir=OUTPUT_DIR):
    """
    Download the 10-K document.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Already downloaded: {filename}")
        return True
    
    try:
        print(f"Downloading: {filename}")
        response = requests.get(doc_url, headers=HEADERS)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        print(f"Successfully downloaded: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return False


def main():
    """
    Download the 5 most recent 10-K filings from multiple companies.
    """
    # Well-known company CIKs for getting recent 10-K filings
    test_companies = [
        {"name": "Apple Inc.", "cik": 320193},
        {"name": "Microsoft Corp", "cik": 789019},
        {"name": "Tesla Inc", "cik": 1318605},
        {"name": "Amazon.com Inc", "cik": 1018724},
        {"name": "Alphabet Inc", "cik": 1652044},
    ]
    
    print("Downloading the 5 most recent SEC 10-K filings...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    downloaded_count = 0
    target_count = 5
    
    for company in test_companies:
        if downloaded_count >= target_count:
            break
            
        print(f"\nProcessing {company['name']} (CIK: {company['cik']})")
        
        # Get company filings
        filings_data = get_company_filings(company["cik"])
        if not filings_data:
            continue
        
        # Find recent 10-K filings (limit to what we still need)
        remaining_needed = target_count - downloaded_count
        ten_k_filings = find_recent_10k_filings(filings_data, limit=remaining_needed)
        
        if not ten_k_filings:
            print(f"No recent 10-K filings found for {company['name']}")
            continue
        
        print(f"Found {len(ten_k_filings)} recent 10-K filing(s)")
        
        for filing in ten_k_filings:
            if downloaded_count >= target_count:
                break
                
            print(f"  Filing date: {filing['filingDate']}")
            print(f"  Accession number: {filing['accessionNumber']}")
            
            # Construct filing detail URL
            detail_url = construct_filing_url(filing["cik"], filing["accessionNumber"])
            print(f"  Detail URL: {detail_url}")
            
            # Get the 10-K document URL
            doc_url = get_10k_document_url(detail_url)
            if doc_url:
                print(f"  Document URL: {doc_url}")
                
                # Create filename
                filename = f"{company['name'].replace(' ', '_')}_{filing['filingDate']}_10K.html"
                filename = filename.replace(',', '').replace('.', '')
                
                # Download the document
                if download_document(doc_url, filename):
                    downloaded_count += 1
                    print(f"  Progress: {downloaded_count}/{target_count} files downloaded")
            else:
                print(f"  Could not find 10-K document URL")
            
            # Rate limiting
            time.sleep(0.1)
        
        # Longer pause between companies
        time.sleep(1.0)
    
    print(f"\nCompleted! Downloaded {downloaded_count} 10-K filings to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
