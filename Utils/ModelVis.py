"""
Model Visualisations
"""

# Imports
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

from keras.utils.vis_utils import plot_model

from keras_visualizer import visualizer

import visualkeras
from PIL import ImageFont

# Main Functions
# Model Functions
def Model_LoadModel(path):
    '''
    Load Model    
    '''
    model = load_model(path)
    return model

# Visualisation Functions
# Keras
def ModelVis_Keras_SequentialModel_BlockView(
    model, 
    save_path="model.png", dpi=96, display=True, 
    show_shapes=False, show_dtype=False, show_layer_names=True, show_layer_activations=False,
    expand_nested=False, 
    **params
    ):
    '''
    Visualise Sequential Model - Block View
    '''
    # Set Params
    vis_params = {
        "to_file": save_path,
        "dpi": dpi, 

        "show_shapes": show_shapes, 
        "show_dtype": show_dtype, 
        "show_layer_names": show_layer_names, 
        "show_layer_activations": show_layer_activations, 
        "expand_nested": expand_nested, 
    }

    # Visualise
    plot_model(model, **vis_params)
    I_vis = cv2.imread(save_path)

    # Display
    if display:
        plt.imshow(I_vis)
        plt.show()

    return I_vis

# Keras Visualiser
def ModelVis_KerasVisualizer_SequentialModel(
    model, 
    save_path=None, display=True, 
    **params
    ):
    '''
    Visualise Sequential Model - Simple
    '''
    # Set Params
    vis_params = {
        "view": display,
        "format": "png"
    }
    if save_path is not None: vis_params["filename"] = save_path

    # Visualise
    I_vis = visualizer(model, **vis_params)

    return I_vis

# VisualKeras
def ModelVis_VisualKeras_SequentialModel_LayerView(
    model, 
    save_path=None, display=True, 
    **params
    ):
    '''
    Visualise Sequential Model - Layered View
    '''
    # Set Params
    vis_params = {
        "legend": True,
        "font": ImageFont.load_default(),
        "draw_volume": True
    }
    if save_path is not None: vis_params["to_file"] = save_path

    # Visualise
    I_vis = visualkeras.layered_view(model, **vis_params)

    # Display
    if display:
        plt.imshow(I_vis)
        plt.show()

    return I_vis

def ModelVis_VisualKeras_SequentialModel_GraphView(
    model, 
    save_path=None, display=True, 
    **params
    ):
    '''
    Visualise Sequential Model - Graph View
    '''
    # Set Params
    vis_params = {
        
    }
    if save_path is not None: vis_params["to_file"] = save_path

    # Visualise
    I_vis = visualkeras.graph_view(model, **vis_params)

    # Display
    if display:
        plt.imshow(I_vis)
        plt.show()

    return I_vis