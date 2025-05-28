import tarfile
import os
import argparse


def gzip_folder(args):
    """
    Compresses an entire folder into a single .tar.gz file.
    
    Args:
        :param args:
            --folder_path (str): Path to the folder to compress.
            --output_file (str): Path to the output .tar.gz file.
    """
    with tarfile.open(args.output_file, "w:gz") as tar:
        tar.add(args.folder_path, arcname=os.path.basename(args.folder_path))
    print(f"Folder '{args.folder_path}' has been compressed into '{args.output_file}'")



folder_path = r"C:\Users\Lorenzo\Desktop\submission"
output_file = r"C:\Users\Lorenzo\Desktop\submission.gz"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--folder_path", default = folder_path, type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--output_file",  default = output_file, type=str, help="Path to the training dataset (optional).")
    args_ = parser.parse_args()
    gzip_folder(args_)

    #folder_path = "/Users/giuseppedaddario/Desktop/submission"
    #output_file = "/Users/giuseppedaddario/Desktop/submission.gz"


