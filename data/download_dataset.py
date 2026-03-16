import os
import urllib.request
import zipfile

import os
import urllib.request
import zipfile
import subprocess

def download_and_extract(url, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "dataset.zip")
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully.")
    except Exception as e:
        print(f"Error extracting dataset: {e}")

def download_kaggle_dataset(dataset_name, extract_to):
    print(f"Attempting to download {dataset_name} via Kaggle API...")
    os.makedirs(extract_to, exist_ok=True)
    try:
        # Requires kaggle CLI installed and ~/.kaggle/kaggle.json configured
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", extract_to, "--unzip"], check=True)
        print(f"Successfully downloaded {dataset_name}")
    except FileNotFoundError:
        print("Kaggle CLI not found. Please run 'pip install kaggle' and configure your API key (kaggle.json).")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {dataset_name}. Ensure your kaggle.json is set up. Error: {e}")

if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. GTSDB (German Traffic Sign Detection Benchmark - Direct Link)
    gtsdb_url = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"
    download_and_extract(gtsdb_url, os.path.join(data_dir, "gtsdb"))
    