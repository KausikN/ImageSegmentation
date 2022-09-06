"""
Image Segmentation - Histogram
"""

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Utils Functions
def Dist_Normal(x, mean=0.0, std=1.0):
    '''
    Dist - Normal Distribution
    '''
    var = max(std**2, 0.00001)
    return np.exp(-((x - mean)**2) / (2 * var))

# Main Functions
def ImageSegmentation_Histogram_MeanMax(
    I, 

    grayscale=True,
    hist_ref_multiplier=0.5,
    hist_ref_range_multiplier=0.5,
    N_BINS=256,
    hard_segmentation=-True,

    visualise=False,
    **params
    ):
    '''
    ImageSegmentation - Basic - Histogram

    Segment image based on value histogram.

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
    bins = np.linspace(0.0, 1.0, N_BINS+1)
    for c in range(I.shape[2]):
        # Get Histogram
        I_c = I[:, :, c]
        I_hist, bin_edges = np.histogram(I_c, bins=bins)
        hist_min_ind = np.argmin(I_hist)
        hist_max_ind = np.argmax(I_hist)
        hist_mid_vals = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        hist_mean = np.sum(hist_mid_vals * I_hist) / np.sum(I_hist)
        # Get Background Probability
        hist_min = hist_mid_vals[hist_min_ind]
        hist_max = hist_mid_vals[hist_max_ind]
        hist_ref = hist_mean*hist_ref_multiplier + hist_max*(1-hist_ref_multiplier)
        hist_ref_range = abs(hist_mean - hist_max)*hist_ref_range_multiplier
        # Add to Segmentation Map
        map_c = Dist_Normal(I_c, hist_ref, hist_ref_range)
        SEG_MAP[:, :, 0] += map_c
        # Record History
        HISTORY.append({
            "I_hist": I_hist,
            "hist_mid_vals": hist_mid_vals,
            "hist_min": hist_min,
            "hist_max": hist_max,
            "hist_mean": hist_mean,
            "hist_ref": hist_ref,
            "hist_ref_range": hist_ref_range,
            "map_c": map_c
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
            "Histogram": [],
            "Probability Map": []
        }
        Data = {
            "Histogram": {}
        }
        for i in range(len(HISTORY)):
            data = HISTORY[i]
            # Init
            maxVal = np.max(data["I_hist"])
            totalFreq = np.sum(data["I_hist"])
            probMap = Dist_Normal(data["hist_mid_vals"], data["hist_ref"], data["hist_ref_range"])
            # Channels Plot
            fig_channel = plt.figure()
            plt.imshow(I[:, :, i], cmap="gray")
            plt.title(f"Channel {i}")
            # Histogram Plot
            fig_hist = plt.figure()
            ## Histogram
            plt.plot(data["hist_mid_vals"], data["I_hist"], label="hist")
            ## Hist min and max
            plt.plot([data["hist_min"], data["hist_min"]], [0, maxVal], label="hist_min")
            plt.plot([data["hist_max"], data["hist_max"]], [0, maxVal], label="hist_max")
            ## Hist mean
            plt.plot([data["hist_mean"], data["hist_mean"]], [0, maxVal], label="hist_mean")
            ## Hist ref
            plt.plot([data["hist_ref"], data["hist_ref"]], [0, maxVal], label="hist_ref")
            ## Hist ref range
            ref_bounds = [
                [data["hist_ref"] - data["hist_ref_range"], data["hist_ref"] + data["hist_ref_range"]],
                [0, maxVal]
            ]
            plt.plot(
                [ref_bounds[0][0], ref_bounds[0][0], ref_bounds[0][1], ref_bounds[0][1], ref_bounds[0][0]], 
                [ref_bounds[1][0], ref_bounds[1][1], ref_bounds[1][1], ref_bounds[1][0], ref_bounds[1][0]], 
                label="hist_bounds"
            )
            ## Other
            plt.legend()
            plt.title(f"Channel {i}")
            plt.xlabel("Value")
            plt.ylabel("Count")
            # Prob Map Plot
            fig_map = plt.figure()
            ## Histogram
            plt.plot(data["hist_mid_vals"], data["I_hist"]/totalFreq, label="freq")
            ## Prob Map
            plt.plot(data["hist_mid_vals"], probMap, label="prob")
            # Threshold
            if hard_segmentation: plt.plot([0, data["hist_mid_vals"][-1]], [0.5, 0.5], label="hard_threshold")
            ## Other
            plt.legend()
            plt.title(f"Channel {i}")
            plt.xlabel("Value")
            plt.ylabel("Freq / Prob")
            # Record
            Plots["Channel"].append(fig_channel)
            Plots["Histogram"].append(fig_hist)
            Plots["Probability Map"].append(fig_map)
            Data["Histogram"][f"Channel {i}"] = {
                "hist_min": data["hist_min"],
                "hist_max": data["hist_max"],
                "hist_mean": data["hist_mean"],
                "hist_ref": data["hist_ref"],
                "hist_ref_range": data["hist_ref_range"]
            }
            # CleanUp
            plt.close(fig_channel)
            plt.close(fig_hist)
            plt.close(fig_map)
        # Record
        VisData["figs"]["pyplot"]["Channels"] = Plots["Channel"]
        VisData["figs"]["pyplot"]["Histogram"] = Plots["Histogram"]
        VisData["figs"]["pyplot"]["Probability Map"] = Plots["Probability Map"]
        VisData["data"]["Histogram"] = Data["Histogram"]

    OutData = {
        "map": SEG_MAP,
        **VisData
    }
    return OutData

# Main Vars
SEG_FUNCS = {
    "Mean-Max": {
        "func": ImageSegmentation_Histogram_MeanMax,
        "params": {
            "grayscale": True,
            "hist_ref_multiplier": 0.5,
            "hist_ref_range_multiplier": 0.5,
            "N_BINS": 256,
            "hard_segmentation": True
        }
    }
}