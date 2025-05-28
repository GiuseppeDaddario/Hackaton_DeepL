import tarfile
import os
import argparse


FOLDER_PATH = r"C:\Users\Lorenzo\Desktop\submission"
OUTPUT_FILE = r"C:\Users\Lorenzo\Desktop\submission.gz"

# FOLDER_PATH = "/Users/giuseppedaddario/Desktop/submission"
# OUTPUT_FILE = "/Users/giuseppedaddario/Desktop/submission.gz"




#################
## COMPRESSION ##
#################

def gzip_folder(args):

##  --folder_path (str): Path to the folder to compress.
##  --output_file (str): Path to the output .tar.gz file.
    
    with tarfile.open(args.output_file, "w:gz") as tar:
        tar.add(args.folder_path, arcname=os.path.basename(args.folder_path))
    print(f"Folder '{args.folder_path}' has been compressed into '{args.output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--folder_path", default = FOLDER_PATH, type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--output_file",  default = OUTPUT_FILE, type=str, help="Path to the training dataset (optional).")
    args_ = parser.parse_args()
    gzip_folder(args_)

###################
##      END      ##
###################

