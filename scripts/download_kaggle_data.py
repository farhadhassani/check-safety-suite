import kagglehub
import shutil
import os

def download_bcsd():
    print("Downloading BCSD dataset...")
    # Download latest version
    path = kagglehub.dataset_download("saifkhichi96/bank-checks-signatures-segmentation-dataset")
    print("Path to dataset files:", path)
    
    # Move to data directory
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bcsd")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # Copy files
    # kagglehub downloads to a cache dir. We want to copy/move them to our data dir for the project.
    # The path returned is the directory containing the dataset.
    if os.path.exists(path):
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print(f"Dataset copied to {target_dir}")
    else:
        print("Download failed or path not found.")

if __name__ == "__main__":
    download_bcsd()
