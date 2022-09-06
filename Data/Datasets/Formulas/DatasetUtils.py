"""
Dataset Utils for Formulas Dataset

Expected Files in Dataset Folder:
    - Images/              :  All input images are saved in this folder
    - Maps/                :  All segmentation maps are saved in this folder
    - formulas_test.csv    :  Path of image and segementation map are saved in this file
"""

# Imports
import io
import os
import cv2
import zipfile
import functools
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Main Vars
DATASET_PATH = "Data/Datasets/Formulas/TestData/"
DATASET_ITEMPATHS = {
    "images": "Images/",
    "map_images": "Maps/",
    "test": "formulas_test.csv"
}

# Main Functions
# Load Functions
def DatasetUtils_LoadCSV(path):
    '''
    DatasetUtils - Load CSV
    '''
    return pd.read_csv(path)

# Dataset Functions
def DatasetUtils_LoadDataset(path=DATASET_PATH, mode="test", N=-1, DATASET_ITEMPATHS=DATASET_ITEMPATHS, **params):
    '''
    DatasetUtils - Load Formulas Dataset
    Pandas Dataframe with columns:
        - "path" - Path to image : Input X
        - "map_path" - Path to Segementation Map : Input Y
    '''
    # Get Dataset
    dataset_info = DatasetUtils_LoadCSV(os.path.join(path, DATASET_ITEMPATHS[mode]))
    # Take N range
    if type(N) == int:
        if N > 0: dataset_info = dataset_info.head(N)
    elif type(N) == list:
        if len(N) == 2: dataset_info = dataset_info.iloc[N[0]:N[1]]
    # Reset Columns
    dataset_info.columns = ["path", "map_path"]
    # Add Main Path
    dataset_info["path"] = dataset_info["path"].apply(lambda x: os.path.join(path, DATASET_ITEMPATHS["images"], x))
    dataset_info["map_path"] = dataset_info["map_path"].apply(lambda x: os.path.join(path, DATASET_ITEMPATHS["map_images"], x))

    return dataset_info

# Main Vars
DATASET_FUNCS = {
    "full": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "train": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "val": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS),
    "test": functools.partial(DatasetUtils_LoadDataset, mode="test", DATASET_ITEMPATHS=DATASET_ITEMPATHS)
}