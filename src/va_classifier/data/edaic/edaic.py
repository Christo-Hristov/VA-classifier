"""
download_edaicwoz.py - E-DAIC Dataset Download and Preprocessing
==============================================================

This script downloads and extracts transcript files from the E-DAIC dataset (Extended Distress
Analysis Interview Corpus - Wizard of Oz). E-DAIC contains clinical interviews with participants
discussing their mental health, annotated with PHQ-8 depression scores and PCL-5 PTSD scores.

Key processing steps:
1. Scrape the E-DAIC data portal for available transcript archives
2. Download .tar.gz files containing participant interview data
3. Extract only the Transcript.csv files from each archive
4. Organize transcripts in a single directory for further processing
5. Clean up temporary files

Dataset: https://dcapswoz.ict.usc.edu/wwwedaic/
Paper: "The Extended Distress Analysis Interview Corpus (E-DAIC): Natural Conversational 
       Language for Mental Health" (Gratch et al., 2014)

Note: This dataset requires access approval and contains sensitive mental health data.
      Ensure proper ethical guidelines and data handling protocols are followed.

Usage:
    python download_edaicwoz.py
    
Output:
    - Raw transcripts saved to: data/processed/edaic_transcripts/
    - Each file named: [ParticipantID]_Transcript.csv
"""

import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import tarfile
import shutil

# Configuration
INDEX_URL = "https://dcapswoz.ict.usc.edu/wwwedaic/data/"
DOWNLOAD_DIR = "data/edaic_raw"
EXTRACT_DIR = DOWNLOAD_DIR
TRANSCRIPT_DEST = "data/processed/edaic_transcripts"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DEST, exist_ok=True)

def get_tar_links(index_url, start_idx=234):
    resp = requests.get(index_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = [
        a['href'] for a in soup.find_all('a', href=True)
        if a['href'].endswith("_P.tar.gz")
    ]
    return sorted(links)[start_idx:]

def download_file(url, dest_path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for chunk in resp.iter_content(1024):
            f.write(chunk)
            bar.update(len(chunk))

def extract_transcript_csv(tar_path, extract_dir, output_dir):
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("Transcript.csv"):
                    tar.extract(member, extract_dir)
                    subfolder = os.path.dirname(member.name)
                    source = os.path.join(extract_dir, member.name)
                    target = os.path.join(output_dir, os.path.basename(member.name))
                    shutil.move(source, target)
                    print(f"✅ Extracted {os.path.basename(member.name)}")
    except Exception as e:
        print(f"❌ Error extracting {tar_path}: {e}")

def main():
    links = get_tar_links(INDEX_URL)
    for link in links:
        file_url = INDEX_URL + link
        local_path = os.path.join(DOWNLOAD_DIR, link)
        
        try:
            download_file(file_url, local_path)
            extract_transcript_csv(local_path, EXTRACT_DIR, TRANSCRIPT_DEST)
            os.remove(local_path)
        except Exception as e:
            print(f"⚠️ Skipped {link} due to error: {e}")

    print("🎉 All transcript CSVs extracted to:", TRANSCRIPT_DEST)

if __name__ == "__main__":
    main()