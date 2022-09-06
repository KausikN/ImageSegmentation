"""
Image Segmentation - Cluster
"""

# Imports
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Utils Functions
def Dist_Normal(x, mean=0.0, std=1.0):
    '''
    Dist - Normal Distribution
    '''
    var = max(std**2, 0.00001)
    return np.exp(-((x - mean)**2) / (2 * var))

# Main Functions
def ImageSegmentation_Cluster_KMeans(
    I, 

    grayscale=False,
    N_CLUSTERS=2,
    hard_segmentation=-True,

    visualise=False,
    **params
    ):
    '''
    ImageSegmentation - Cluster - KMeans Clustering

    Segment image based on KMeans clustered map.

    Inputs:
        - I : Image (Height, Width, Channels)
            - Values in range [0, 1]
    Outputs:
        - map : Segmentation Map of Image (Height, Width, Classes)
            - Classes: [...]
    '''
    # Init
    N_CLASSES = N_CLUSTERS
    SEG_MAP = np.zeros((I.shape[0], I.shape[1], N_CLASSES))
    # Grayscale
    if grayscale:
        I = np.mean(I, axis=-1)
        I = np.expand_dims(I, axis=-1)
    # Segment for Each Channel to get background probability
    HISTORY = {}
    ## Get Clusters
    I_flat = I.reshape((-1, I.shape[-1]))
    kmeans_data = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(I_flat)
    I_clusterlabels = kmeans_data.labels_
    I_clustered = kmeans_data.cluster_centers_[kmeans_data.labels_]
    I_clusterlabels = np.array(I_clusterlabels)
    I_clustered = I_clustered.reshape(I.shape)
    SEG_MAP = SEG_MAP.reshape((-1, N_CLASSES))
    SEG_MAP[np.arange(SEG_MAP.shape[0]), I_clusterlabels] = 1.0
    SEG_MAP = SEG_MAP.reshape((I.shape[0], I.shape[1], N_CLASSES))
    ## Record
    HISTORY = {
        "I_clustered": I_clustered
    }
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
            "Cluster Map": []
        }
        Data = {}
        # Init
        data = HISTORY
        # Channels Plot
        fig_channel = plt.figure()
        plt.imshow(I, cmap="gray")
        plt.title("Image")
        # Cluster Map Plot
        fig_map = plt.figure()
        plt.imshow(data["I_clustered"], cmap="gray")
        plt.title("Cluster Map")
        # Record
        Plots["Channel"].append(fig_channel)
        Plots["Cluster Map"].append(fig_map)
        # CleanUp
        plt.close(fig_channel)
        plt.close(fig_map)
        # Record
        VisData["figs"]["pyplot"]["Channels"] = Plots["Channel"]
        VisData["figs"]["pyplot"]["Cluster Map"] = Plots["Cluster Map"]

    OutData = {
        "map": SEG_MAP,
        **VisData
    }
    return OutData

# Main Vars
SEG_FUNCS = {
    "KMeans": {
        "func": ImageSegmentation_Cluster_KMeans,
        "params": {
            "grayscale": False,
            "N_CLUSTERS": 2,
            "hard_segmentation": True
        }
    }
}