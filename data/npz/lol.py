import urllib.request
import os
import zipfile
import numpy as np

# Set the download URL and destination directory
url = 'https://ogb.stanford.edu/docs/graphprop/ppi.zip'
download_dir = './ppi_new/'

# Create the download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

    # Download the file
    zip_path, _ = urllib.request.urlretrieve(url, os.path.join(download_dir, 'ppi.zip'))

    # Extract the zip file to the same directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
            
        # Load the dataset

        data = np.load(os.path.join(download_dir, 'ppi.npz'))
        x = data['x']
        y = data['y']
        edge_index = data['edge_index']
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        test_mask = data['test_mask']

        print(f'x shape: {x.shape}')
        print(f'y shape: {y.shape}')
        print(f'edge_index shape: {edge_index.shape}')
        print(f'train_mask shape: {train_mask.shape}')
        print(f'val_mask shape: {val_mask.shape}')
        print(f'test_mask shape: {test_mask.shape}')

