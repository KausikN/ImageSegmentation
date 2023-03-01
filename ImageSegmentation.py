"""
Image Segmentation

Inputs:
 - I : Image (Height, Width, Channels)
    - Values for each channel in each pixel
    - Values in range [0, 1]
Outputs:
 - I_map : Segmentation Map (Height, Width, Classes)
    - Probability values for each class in each pixel
    - Values in range [0, 1]
"""

# Imports
# Segmenter Imports
from SegmentationMethods.Basic import ImageSegmentation_Histogram
from SegmentationMethods.Basic import ImageSegmentation_Filters
from SegmentationMethods.Basic import ImageSegmentation_Cluster
# Dataset Imports
from Data.Datasets.Formulas import DatasetUtils as DatasetUtils_Formulas

# Main Functions
def SegmentationClasses_Default(N_CLASSES):
    '''
    Segmentation Classes - Default
    '''
    if N_CLASSES == 2:
        return ["Background", "Foreground"]
    else:
        return ["Class_" + str(i) for i in range(N_CLASSES)]

# Main Vars
SEGMENTATION_MODULES = {
    "Histogram": ImageSegmentation_Histogram.SEG_FUNCS,
    "Filters": ImageSegmentation_Filters.SEG_FUNCS,
    "Cluster": ImageSegmentation_Cluster.SEG_FUNCS
}

DATASETS = {
    "Formulas": DatasetUtils_Formulas
}