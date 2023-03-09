"""
Image Utils
"""

# Imports
import io
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Main Functions
# Load/Save Functions
def ImageUtils_LoadImage(path):
    '''
    Load Image
    '''
    I = cv2.imread(path)
    I = np.array(I, dtype=float) / 255.0

    return I

def ImageUtils_SaveImage(I, path):
    '''
    Save Image
    '''
    if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
    cv2.imwrite(path, I)

def ImageUtils_LoadMap(path):
    '''
    Load Map
    '''
    I = np.load(path)

    return I

def ImageUtils_SaveMap(I, path):
    '''
    Save Map
    '''
    np.save(path, I)

# Conversion Functions
def ImageUtils_Bytes2Array(I_bytes):
    '''
    Image Bytes to Array
    '''
    # Load Image
    I_array = np.array(Image.open(io.BytesIO(I_bytes)), dtype=float)
    # print("Utils:", I_array.shape, I_array.dtype, I_array.min(), I_array.max())
    # Fix Dims
    if I_array.ndim == 2: I_array = np.dstack((I_array, I_array, I_array))
    elif I_array.shape[2] > 3: I_array = I_array[:, :, :3]
    # Normalize
    if I_array.max() > 1.0: I_array /= 255.0

    return I_array

def ImageUtils_Array2Bytes(I):
    '''
    Array to Bytes
    '''
    I = np.array(I * 255.0, dtype=np.uint8)
    I_bytes = I.tobytes()

    return I_bytes

# Plot Functions
def ImageUtils_PlotImageHistogram(I, bins=256):
    '''
    Plot Image Histogram
    '''
    # Convert Values
    vals = I.ravel()
    # Plot Histogram
    fig = plt.figure()
    plt.hist(vals, bins=bins, range=(I.min(), I.max()))
    plt.title("Image Histogram")

    return fig

# Visualisation Functions
def ImageVis_SegmentationMap(I, I_map, classes, cmap="jet"):
    '''
    Visualise Segmentation Map
    '''
    # Init
    N_CLASSES = I_map.shape[-1]
    # Init Figure
    n_cols = 2
    n_rows = N_CLASSES
    fig = plt.figure()
    # Plot
    for i in range(N_CLASSES):
        # Mask
        plt.subplot(n_rows, n_cols, (2*i)+1)
        plt.imshow(I_map[:, :, i], cmap=cmap)
        plt.title(f"Class {classes[i]}")
        plt.axis("off")
        # Image
        I_masked = I * I_map[:, :, i].reshape((I_map.shape[0], I_map.shape[1], 1))
        plt.subplot(n_rows, n_cols, (2*i)+2)
        plt.imshow(I_masked)
        plt.title(f"Masked {classes[i]}")
        plt.axis("off")
    plt.close(fig)

    return fig

# Clean Functions
def ImageUtils_Clean(I):
    '''
    Clean Image
     - Normalise
     - Change Background to Black
    '''
    # Normalise
    I_normalised = (I - I.min()) / (I.max() - I.min())
    # Flip Background Color to Black Always
    # If I_mean is closer to I_max than I_min, then flip as background is currently I_max, i.e. white
    flip = (1.0 - I_normalised.mean()) < (I_normalised.mean() - 0.0)
    if flip: I_normalised = 1.0 - I_normalised

    I_cleaned = I_normalised

    return I_cleaned

def ImageUtils_Resize(I, maxSize=1024):
    '''
    Resize Image to have max width or height as given
    '''
    aspect_ratio = I.shape[1] / I.shape[0]
    newSize = (maxSize, maxSize)
    if aspect_ratio > 1:
        newSize = (int(maxSize / aspect_ratio), maxSize)
    else:
        newSize = (maxSize, int(maxSize * aspect_ratio))
    newSize = newSize[::-1]

    I_resized = cv2.resize(I, newSize)

    return I_resized

# Padding Functions
def ImageUtils_Pad(I, padSizes=[0, 0, 0, 0], padValue=0.0):
    '''
    Pad Image: Top, Bottom, Left, Right
    '''
    # Pad
    I_FinalShape = (I.shape[0] + padSizes[0] + padSizes[1], I.shape[1] + padSizes[2] + padSizes[3])
    I_padded = np.ones(I_FinalShape, dtype=I.dtype) * padValue
    I_padded[padSizes[0]:padSizes[0]+I.shape[0], padSizes[2]:padSizes[2]+I.shape[1]] = I

    return I_padded

# Effect Functions
def ImageUtils_Effect_InvertColor(I):
    '''
    Invert Image Values
    '''
    I_flipped = 1.0 - I

    return I_flipped

def ImageUtils_Effect_Normalise(I):
    '''
    Normalise Image
    '''
    I_normalised = (I - I.min()) / (I.max() - I.min())

    return I_normalised

def ImageUtils_Effect_Binarise(I, threshold=0.5):
    '''
    Binarise Image
    '''
    I_binarised = I > threshold
    I_binarised = np.array(I_binarised, dtype=float)

    return I_binarised

def ImageUtils_Effect_Sharpen(I):
    '''
    Sharpen Image
    '''
    # Sharpen
    KERNEL_SHARPEN_1 = np.array([
        [0., -1/5, 0.],
        [-1/5, 1, -1/5],
        [0., -1/5, 0.]
    ])
    I_shapened = cv2.filter2D(src=I, ddepth=-1, kernel=KERNEL_SHARPEN_1)
    # Clip
    I_shapened = np.clip(I_shapened, 0.0, 1.0)

    return I_shapened

def ImageUtils_Effect_Erode(I, iterations=1):
    '''
    Erode Image
    '''
    # Erode
    KERNEL_ERODE_1 = np.ones((3, 3))
    I_eroded = cv2.erode(I, KERNEL_ERODE_1, iterations=iterations)

    return I_eroded