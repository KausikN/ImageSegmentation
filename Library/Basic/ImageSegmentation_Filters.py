"""
Image Segmentation - Filters
"""

# Imports
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Main Functions
def ImageSegmentation_Filters_Laplace(
    I, 

    grayscale=True,
    normalize_edge_map=True,
    hard_segmentation=-True,

    visualise=False,
    **params
    ):
    '''
    ImageSegmentation - Filters - Laplace Filtering

    Segment image based on laplace filtered edge map.

    Inputs:
        - I : Image (Height, Width, Channels)
            - Values in range [0, 1]
    Outputs:
        - map : Segmentation Map of Image (Height, Width, Classes=2)
            - Classes: [Background, Foreground]
    '''
    # Init
    N_CLASSES = 2
    SEG_MAP = np.zeros((I.shape[0], I.shape[1], N_CLASSES))
    # Grayscale
    if grayscale:
        I = np.mean(I, axis=-1)
        I = np.expand_dims(I, axis=-1)
    # Segment for Each Channel to get background probability
    HISTORY = []
    for c in range(I.shape[2]):
        # Get Edge Map
        I_c = I[:, :, c]
        I_edge = ndimage.laplace(I_c)
        # Normalise
        if normalize_edge_map:
            I_edge = (I_edge - np.min(I_edge)) / (np.max(I_edge) - np.min(I_edge))
        # Get Background Probability
        I_bgprob = 1.0 - I_edge
        # Add to Segmentation Map
        map_c = I_bgprob
        SEG_MAP[:, :, 0] += map_c
        # Record History
        HISTORY.append({
            "I_edge": I_edge
        })
    # Divide by number of channels to get background probability
    SEG_MAP[:, :, 0] /= I.shape[2]
    # Get Foreground Probability
    SEG_MAP[:, :, 1] = 1.0 - SEG_MAP[:, :, 0]
    # Threshold
    if hard_segmentation:
        maxClasses = np.argmax(SEG_MAP, axis=-1)
        SEG_MAP[:, :, :] = 0.0
        for i in range(N_CLASSES): SEG_MAP[maxClasses == i, i] = 1.0

    # Visualise
    VisData = {
        "figs": {
            "pyplot": {},
            "plotly_chart": {}
        },
        "data": {}
    }
    if visualise:
        Plots = {
            "Channel": [],
            "Edge Map": []
        }
        Data = {}
        for i in range(len(HISTORY)):
            data = HISTORY[i]
            # Init
            
            # Channels Plot
            fig_channel = plt.figure()
            plt.imshow(I[:, :, i], cmap="gray")
            plt.title(f"Channel {i}")
            # Edge Map Plot
            fig_map = plt.figure()
            plt.imshow(data["I_edge"], cmap="gray")
            plt.title(f"Edge Map {i}")
            # Record
            Plots["Channel"].append(fig_channel)
            Plots["Edge Map"].append(fig_map)
            # CleanUp
            plt.close(fig_channel)
            plt.close(fig_map)
        # Record
        VisData["figs"]["pyplot"]["Channels"] = Plots["Channel"]
        VisData["figs"]["pyplot"]["Edge Map"] = Plots["Edge Map"]

    OutData = {
        "map": SEG_MAP,
        **VisData
    }
    return OutData

# Main Vars
SEG_FUNCS = {
    "Laplace": {
        "func": ImageSegmentation_Filters_Laplace,
        "params": {
            "grayscale": True,
            "normalize_edge_map": True,
            "hard_segmentation": True
        }
    }
}