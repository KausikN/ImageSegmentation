"""
Streamlit App - Dataset
"""

# Imports
import os
import cv2
import json
import streamlit as st
from tqdm import tqdm

from ImageSegmentation import *
from Utils.ImageUtils import *

# Main Vars
TEMP_PATH = "Data/Temp/"

DEFAULT_CMAP = "gray"

# Plot Functions
def Plot_UniqueValCounts(data, title=""):
    '''
    Plot the Unique value counts as bar chart
    '''
    fig = plt.figure()
    data_unique, data_unique_counts = np.unique(data, return_counts=True)
    sorted_indices = np.argsort(data_unique)
    plt.bar(data_unique[sorted_indices], data_unique_counts[sorted_indices])
    plt.title(title)
    plt.close(fig)

    return fig

# UI Functions
def UI_LoadDataset():
    '''
    Load Dataset
    '''
    st.markdown("## Load Dataset")
    # Select Dataset
    USERINPUT_Dataset = st.selectbox("Select Dataset", list(DATASETS.keys()))
    DATASET_MODULE = DATASETS[USERINPUT_Dataset]
    # Load Dataset
    DATASET = DATASET_MODULE.DATASET_FUNCS["test"]()
    N = DATASET.shape[0]

    # Subset Dataset
    st.markdown("## Subset Dataset")
    col1, col2 = st.columns(2)
    USERINPUT_DatasetStart = col1.number_input("Subset Dataset (Start Index)", 0, N-1, 0)
    USERINPUT_DatasetCount = col2.number_input("Subset Dataset (Count)", 1, N, N)
    USERINPUT_DatasetCount = min(USERINPUT_DatasetCount, N-USERINPUT_DatasetStart)
    DATASET = DATASET.iloc[USERINPUT_DatasetStart:USERINPUT_DatasetStart+USERINPUT_DatasetCount]
    DATASET.reset_index(drop=True, inplace=True)

    # Display
    N = DATASET.shape[0]
    USERINPUT_ViewSampleIndex = st.slider(f"View Sample ({N} Samples)", 0, N-1, 0, 1)
    st.image(DATASET["path"][USERINPUT_ViewSampleIndex], caption=f"Image: {USERINPUT_ViewSampleIndex}", use_column_width=True)
    SEG_MAP = ImageUtils_LoadMap(DATASET["map_path"][USERINPUT_ViewSampleIndex])
    st.pyplot(ImageVis_SegmentationMap(
        SEG_MAP,
        classes=SegmentationClasses_Default(SEG_MAP.shape[2]),
        cmap=DEFAULT_CMAP
    ))

    return DATASET

def UI_VisualiseDatasetImages(DATASET):
    '''
    Visualisations on Dataset Images
    '''
    # Load Images Info
    IMAGES_SIZES = []
    N = DATASET.shape[0]
    progressObj = st.progress(0.0)
    for i in tqdm(range(N)):
        I = np.array(cv2.imread(DATASET["path"][i]))
        IMAGES_SIZES.append((I.shape[0], I.shape[1]))
        progressObj.progress((i+1) / N)
    IMAGES_SIZES = np.array(IMAGES_SIZES)
    ASPECT_RATIOS = IMAGES_SIZES[:, 0] / IMAGES_SIZES[:, 1]
    # Images Plot Visualisations
    HEIGHTS_FIG = Plot_UniqueValCounts(IMAGES_SIZES[:, 0], title="Heights")
    WIDTHS_FIG = Plot_UniqueValCounts(IMAGES_SIZES[:, 1], title="Widths")
    ASPECT_RATIOS_FIG = plt.figure()
    plt.hist(ASPECT_RATIOS, bins=100)
    plt.title("Aspect Ratios")
    # Display
    IMAGES_VIS = {
        "Num Images": N,
        "Height": {
            "Min": int(np.min(IMAGES_SIZES[:, 0])),
            "Max": int(np.max(IMAGES_SIZES[:, 0])),
            "Mean": np.mean(IMAGES_SIZES[:, 0]),
            "Median": np.median(IMAGES_SIZES[:, 0]),
            "Std": np.std(IMAGES_SIZES[:, 0])
        },
        "Width": {
            "Min": int(np.min(IMAGES_SIZES[:, 1])),
            "Max": int(np.max(IMAGES_SIZES[:, 1])),
            "Mean": np.mean(IMAGES_SIZES[:, 1]),
            "Median": np.median(IMAGES_SIZES[:, 1]),
            "Std": np.std(IMAGES_SIZES[:, 1])
        },
        "Aspect Ratio": {
            "Min": np.min(ASPECT_RATIOS),
            "Max": np.max(ASPECT_RATIOS),
            "Mean": np.mean(ASPECT_RATIOS),
            "Median": np.median(ASPECT_RATIOS),
            "Std": np.std(ASPECT_RATIOS)
        }
    }
    st.markdown("### Images")
    st.write(IMAGES_VIS)
    st.plotly_chart(HEIGHTS_FIG)
    st.plotly_chart(WIDTHS_FIG)
    st.plotly_chart(ASPECT_RATIOS_FIG)
    # CleanUp
    plt.close(ASPECT_RATIOS_FIG)

def UI_VisualiseDataset(DATASET):
    '''
    Standard Visualisations on Dataset
    '''
    st.markdown("## Visualisations")

    # Images Visualisations
    if st.button("Visualise Images"):
        UI_VisualiseDatasetImages(DATASET)

# Mode Functions
def visualise_dataset():
    # Title
    st.markdown("# Visualise Dataset")

    # Load Inputs
    USERINPUT_Dataset = UI_LoadDataset()

    # Visualise Dataset
    UI_VisualiseDataset(USERINPUT_Dataset)

# Main Vars
MODES = {
    "Visualise Dataset": visualise_dataset
}

# Main Functions
def app_main():
    # Title
    st.markdown("# Image Segmentation Dataset Utils")

    # Load Inputs
    # Method
    USERINPUT_Mode = st.sidebar.selectbox("Select Mode", list(MODES.keys()))
    USERINPUT_ModeFunc = MODES[USERINPUT_Mode]
    USERINPUT_ModeFunc()


# RunCode
if __name__ == "__main__":
    # Assign Objects
    
    # Run Main
    app_main()