import wget
import os
import tarfile
import gzip
import zipfile
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--glove", action="store_true")
args = parser.parse_args()

# Extract data file
with tarfile.open("summary.tar.gz", "r:gz") as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar)

with gzip.open("sumdata/train/train.article.txt.gz", "rb") as gz:
    with open("sumdata/train/train.article.txt", "wb") as out:
        out.write(gz.read())

with gzip.open("sumdata/train/train.title.txt.gz", "rb") as gz:
    with open("sumdata/train/train.title.txt", "wb") as out:
        out.write(gz.read())

if args.glove:
    glove_dir = "glove"
    glove_url = "https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip"

    if not os.path.exists(glove_dir):
        os.mkdir(glove_dir)

    # Download glove vector
    wget.download(glove_url, out=glove_dir)

    # Extract glove file
    with zipfile.ZipFile(os.path.join("glove", "glove.42B.300d.zip"), "r") as z:
        z.extractall(glove_dir)
