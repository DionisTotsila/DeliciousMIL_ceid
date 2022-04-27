import requests
import zipfile
import os
from os.path import exists

zip_path = 'data/DeliciousMIL.zip'

all_files = [
             'labeled_test_sentences.dat', 
             'labels.txt',
             'test-data.dat',
             'test-label.dat',
             'test-sentlabel.dat',
             'train-data.dat',
             'train-label.dat',
             'vocabs.txt'
            ]

# download zip from url
def download_zip(url, save_as_path):
    r = requests.get(url, allow_redirects=True)
    open(save_as_path, 'wb').write(r.content)

# chech if list of file names exists in data dir, return false if they don't
def file_check(file_list):
    for i in range(len(file_list)):
        if not exists(os.path.join('data/', file_list[i])):
            return False
    return True

def extract_organize(zip_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
    print("Organizing Folders...")
    os.system('mv  data/Data/* data/')
    os.system('rm -r data/Data/')
    os.system('rm ' + zip_path)

if (file_check(all_files)):
    print("All files are ok")
elif(not exists(zip_path)):
    print("Downlading dataset...")
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/00418/DeliciousMIL.zip'
    download_zip(url, 'data/DeliciousMIL.zip')
    extract_organize(zip_path)
else:
    extract_organize(zip_path)