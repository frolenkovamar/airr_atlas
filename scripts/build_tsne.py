import os
import sys
import argparse
import torch
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("embedding", type=str, help="baseline, prott5, etc")
args = parser.parse_args()


out_dir = "/doctorai/marinafr/2023/airr_atlas/analysis/output/"
#out_dir = '/home/marina/Documents/oslo/airr_atlas/output/'

in_file_path = out_dir + args.embedding + '/'

# List all files in the folder
file_list = os.listdir(in_file_path)

# Iterate through the files
for filename in file_list:
    if 'pt' in filename or 'pkl' in filename or 'npy' in filename:
        output_file = in_file_path + 'TSNE_' + filename.split('.')[0]

        # Load embedding file
        if "pt" in filename:
            data = torch.load(in_file_path+filename).numpy()
        elif "pkl" in filename:
            data = pd.read_pickle(in_file_path+filename).values
        elif "npy" in filename:
            data = np.load(in_file_path+filename)
        print(f"Read {in_file_path+filename} with {data.shape[0]} sequences")

        # Apply t-SNE
        tsne_data = TSNE(random_state=42).fit_transform(data)

        # Create a DataFrame with t-SNE results
        tsne_df = pd.DataFrame(tsne_data)

        # Save the t-SNE result to a pickle file
        tsne_df.to_pickle(f'{output_file}.pkl')
        print(f"Saved {output_file}.pkl with {tsne_df.shape[0]} sequences")
