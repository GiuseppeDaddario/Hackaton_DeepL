## Data Loader Module
# This module is responsible for loading data from a specified file path.
# Includes also preprocessing steps.

import torch


def load_data(file_path):

    print("ok")

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data  