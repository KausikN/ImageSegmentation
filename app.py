"""
Stream lit GUI for hosting ImageSegmentation
"""

# Imports
import os
import streamlit as st
import json
import time

from ImageSegmentation import *
from Utils.VideoUtils import *
from Utils.ImageUtils import *

# Main Vars
config = json.load(open("./StreamLitGUI/UIConfig.json", "r"))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    "Choose one of the following",
        tuple(
            [config["PROJECT_NAME"]] + 
            config["PROJECT_MODES"]
        )
    )
    
    if selected_box == config["PROJECT_NAME"]:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(" ", "_").lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config["PROJECT_NAME"])
    st.markdown("Github Repo: " + "[" + config["PROJECT_LINK"] + "](" + config["PROJECT_LINK"] + ")")
    st.markdown(config["PROJECT_DESC"])

    # st.write(open(config["PROJECT_README"], "r").read())

#############################################################################################################################
# Repo Based Vars
PATHS = {
    "cache": "StreamLitGUI/CacheData/Cache.json",
    "default": {
        "example": {
            "image": "Data/InputImages/Test.jpg"
        }
    },
    "temp": "Data/Temp/"
}
DEFAULT_CMAP = "gray"

# Util Vars
CACHE = {}

# Util Functions
def LoadCache():
    global CACHE
    CACHE = json.load(open(PATHS["cache"], "r"))

def SaveCache():
    global CACHE
    json.dump(CACHE, open(PATHS["cache"], "w"), indent=4)

# Main Functions


# UI Functions
def UI_LoadSegAlgo():
    '''
    Load Algorithm for Segmentation
    '''
    st.markdown("## Load Segmentation Algo")
    # Load Method
    USERINPUT_SegModule = st.selectbox("Select Segmentation Module", list(SEGMENTATION_MODULES.keys()))
    cols = st.columns((1, 3))
    USERINPUT_SegMethod = cols[0].selectbox(
        "Select Segmentation Method",
        list(SEGMENTATION_MODULES[USERINPUT_SegModule].keys())
    )
    USERINPUT_SegMethod = SEGMENTATION_MODULES[USERINPUT_SegModule][USERINPUT_SegMethod]
    # Load Params
    USERINPUT_SegParams_str = cols[1].text_area(
        "Params", 
        value=json.dumps(USERINPUT_SegMethod["params"], indent=8),
        height=200
    )
    USERINPUT_SegParams = json.loads(USERINPUT_SegParams_str)
    USERINPUT_SegMethod = {
        "func": USERINPUT_SegMethod["func"],
        "params": USERINPUT_SegParams
    }

    return USERINPUT_SegMethod

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
    if os.path.exists(DATASET["map_path"][USERINPUT_ViewSampleIndex]):
        I = ImageUtils_Bytes2Array(open(DATASET["path"][USERINPUT_ViewSampleIndex], "rb").read())
        SEG_MAP = ImageUtils_LoadMap(DATASET["map_path"][USERINPUT_ViewSampleIndex])
        st.pyplot(ImageVis_SegmentationMap(
            I, SEG_MAP,
            classes=SegmentationClasses_Default(SEG_MAP.shape[2]),
            cmap=DEFAULT_CMAP
        ))

    return DATASET

def UI_LoadImage():
    '''
    Load Image
    '''
    # Load Image
    st.markdown("## Load Image")
    USERINPUT_LoadType = st.selectbox("Load Type", ["Examples", "Upload File", "Datasets"])
    if USERINPUT_LoadType == "Examples":
        # Load Filenames from Default Path
        EXAMPLES_DIR = os.path.dirname(PATHS["default"]["example"]["image"])
        EXAMPLE_FILES = os.listdir(EXAMPLES_DIR)
        USERINPUT_ImagePath = st.selectbox("Select Example File", EXAMPLE_FILES)
        USERINPUT_ImagePath = os.path.join(EXAMPLES_DIR, USERINPUT_ImagePath)
        USERINPUT_Image = open(USERINPUT_ImagePath, "rb").read()
    elif USERINPUT_LoadType == "Upload File":
        USERINPUT_Image = st.file_uploader("Upload Image", type=["jpg", "png", "PNG", "jpeg", "bmp"])
        if USERINPUT_Image is None: USERINPUT_Image = open(PATHS["default"]["example"]["image"], "rb")
        USERINPUT_Image = USERINPUT_Image.read()
    else:
        # Select Dataset
        USERINPUT_Dataset = st.selectbox("Select Dataset", list(DATASETS.keys()))
        DATASET_MODULE = DATASETS[USERINPUT_Dataset]
        # Load Dataset
        DATASET = DATASET_MODULE.DATASET_FUNCS["test"]()
        # Display
        N = DATASET.shape[0]
        USERINPUT_ViewSampleIndex = st.slider(f"View Sample ({N} Samples)", 0, N-1, 0, 1)
        st.image(DATASET["path"][USERINPUT_ViewSampleIndex], caption=f"Image: {USERINPUT_ViewSampleIndex}", use_column_width=True)
        I = ImageUtils_Bytes2Array(open(DATASET["path"][USERINPUT_ViewSampleIndex], "rb").read())
        SEG_MAP = ImageUtils_LoadMap(DATASET["map_path"][USERINPUT_ViewSampleIndex])
        st.pyplot(ImageVis_SegmentationMap(
            I, SEG_MAP,
            classes=SegmentationClasses_Default(SEG_MAP.shape[2]),
            cmap=DEFAULT_CMAP
        ))
        USERINPUT_Image = open(DATASET["path"][USERINPUT_ViewSampleIndex], "rb").read()

    # Convert to Numpy Array
    USERINPUT_Image = ImageUtils_Bytes2Array(USERINPUT_Image)
    # Max Resize
    USERINPUT_MaxSize = st.sidebar.number_input("Max Size", 1, 2048, 512)
    USERINPUT_Image = ImageUtils_Resize(USERINPUT_Image, USERINPUT_MaxSize)
    # Show Image
    st.image(
        USERINPUT_Image, 
        caption=f"Input Image ({USERINPUT_Image.shape[0]}, {USERINPUT_Image.shape[1]})", 
        use_column_width=True
    )

    return USERINPUT_Image

def UI_ImageEdit(USERINPUT_Image):
    '''
    Clean and Edit Image
    '''
    st.markdown("## Image Edit")
    # Get Image Array
    I = USERINPUT_Image
    # Sharpen
    if st.checkbox("Sharpen Image"):
        I = ImageUtils_Effect_Sharpen(I)
        I = ImageUtils_Effect_Normalise(I)
        st.image(I, caption=f"Sharpened Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Binarise
    if st.checkbox("Binarise Image"):
        USERINPUT_BinariseThreshold = st.slider("Binarise Threshold 1", 0.0, 1.0, 0.1, 0.01)
        I = ImageUtils_Effect_Binarise(I, threshold=USERINPUT_BinariseThreshold)
        st.image(I, caption=f"Binarised Image 1 ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Resize
    if st.checkbox("Erode Image"):
        USERINPUT_ResizeMaxSize = st.number_input("Resize Max Size 1", 1, 2048, 1024, 1)
        I = ImageUtils_Resize(I, maxSize=USERINPUT_ResizeMaxSize)
        st.image(I, caption=f"Resized Image 1 ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
        # Erode
        USERINPUT_Iterations = st.slider("Erosion Iterations", 0, 10, 1, 1)
        I = ImageUtils_Effect_Erode(I, iterations=USERINPUT_Iterations)
        # I = ImageUtils_Effect_Normalise(I)
        st.image(I, caption=f"Eroded Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
        # Resize
        USERINPUT_ResizeMaxSize = st.number_input("Resize Max Size 2", 1, 2048, 1024, 1)
        I = ImageUtils_Resize(I, maxSize=USERINPUT_ResizeMaxSize)
        st.image(I, caption=f"Resized Image 2 ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
        # Binarise
        USERINPUT_BinariseThreshold = st.slider("Binarise Threshold 2", 0.0, 1.0, 0.1, 0.01)
        I = ImageUtils_Effect_Binarise(I, threshold=USERINPUT_BinariseThreshold)
        st.image(I, caption=f"Binarised Image 2 ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Normalise
    if st.checkbox("Normalise Image"):
        I = ImageUtils_Effect_Normalise(I)
        st.image(I, caption=f"Normalised Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Resize
    if st.checkbox("Resize Image"):
        USERINPUT_ResizeMaxSize = st.number_input("Resize Max Size", 1, 2048, 256, 1)
        I = ImageUtils_Resize(I, maxSize=USERINPUT_ResizeMaxSize)
        st.image(I, caption=f"Resized Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # Flip
    USERINPUT_InvertColor = st.checkbox("Invert Image Color")
    if USERINPUT_InvertColor: I = ImageUtils_Effect_InvertColor(I)

    st.markdown("## Final Image")
    print("Input:", I.shape, I.dtype, I.min(), I.max())
    I_final = np.array(I * 255.0, dtype=np.uint8)
    st.image(I_final, caption=f"Final Image ({I.shape[0]}, {I.shape[1]})", use_column_width=True)
    # HIST_PLOT = ImageUtils_PlotImageHistogram(I_final)
    # st.plotly_chart(HIST_PLOT)
    # Save Image to Temp
    TempImagePath = os.path.join(PATHS["temp"], "CleanedImage.png")
    ImageUtils_SaveImage(I_final, TempImagePath)
    USERINPUT_Image = open(TempImagePath, "rb").read()
    USERINPUT_Image = ImageUtils_Bytes2Array(USERINPUT_Image)

    return USERINPUT_Image

def UI_DisplayTestResults(TestData):
    '''
    Display Test Results
    '''
    # Test Results
    st.markdown("## Test Results")
    y_true, y_pred = np.array(TestData["y_true"]), np.array(TestData["y_pred"])
    n_matches, n_samples = 0, 0
    for yt, yp in zip(y_true, y_pred):
        n_matches += int(np.count_nonzero(yt != yp) == 0)
        n_samples += 1
    Results = {
        "Num Matches": n_matches,
        "Accuracy": n_matches / n_samples
    }
    st.write(Results)
    # Test Times
    st.markdown("## Test Times")
    Times = {
        "Min Time": float(np.min(TestData["time_exec"])),
        "Max Time": float(np.max(TestData["time_exec"])),
        "Mean Time": float(np.mean(TestData["time_exec"])),
        "Median Time": float(np.median(TestData["time_exec"])),
        "Std Time": float(np.std(TestData["time_exec"]))
    }
    st.write(Times)
    # Test Images
    st.markdown("## Test Images")
    ImageSizes = {
        "height": {
            "Min": float(np.min(TestData["image_size"]["height"])),
            "Max": float(np.max(TestData["image_size"]["height"])),
            "Mean": float(np.mean(TestData["image_size"]["height"])),
            "Median": float(np.median(TestData["image_size"]["height"])),
            "Std": float(np.std(TestData["image_size"]["height"]))
        },
        "width": {
            "Min": float(np.min(TestData["image_size"]["width"])),
            "Max": float(np.max(TestData["image_size"]["width"])),
            "Mean": float(np.mean(TestData["image_size"]["width"])),
            "Median": float(np.median(TestData["image_size"]["width"])),
            "Std": float(np.std(TestData["image_size"]["width"]))
        }
    }
    st.write(ImageSizes)

def UI_DisplayVisData(OutData):
    '''
    Display Algorithm Visualisation Data
    '''
    st.markdown("# Visualisations")
    st.markdown("## Plots")
    # Graphs
    for k in OutData["figs"]["plotly_chart"].keys():
        st.markdown(f"### {k}")
        cols = st.columns(len(OutData["figs"]["plotly_chart"][k]))
        for i in range(len(OutData["figs"]["plotly_chart"][k])):
            cols[i].plotly_chart(OutData["figs"]["plotly_chart"][k][i])
    # Plots
    for k in OutData["figs"]["pyplot"].keys():
        st.markdown(f"### {k}")
        cols = st.columns(len(OutData["figs"]["pyplot"][k]))
        for i in range(len(OutData["figs"]["pyplot"][k])):
            cols[i].pyplot(OutData["figs"]["pyplot"][k][i])
    # Data
    st.markdown("## Data")
    for k in OutData["data"].keys():
        st.markdown(f"### {k}")
        st.write(OutData["data"][k])

# Repo Based Functions
def single_image_segmentation():
    # Title
    st.markdown("# Segmentation - Single Image")

    # Load Inputs
    # Image
    USERINPUT_Image = UI_LoadImage()
    # Clean and Edit Image
    # USERINPUT_Image = UI_ImageEdit(USERINPUT_Image)
    # Method
    USERINPUT_SegMethod = UI_LoadSegAlgo()

    # Process Inputs
    USERINPUT_Process = st.checkbox("Stream Process")
    if not USERINPUT_Process: USERINPUT_Process = st.button("Run Segmentation")
    if USERINPUT_Process:
        # Segementation
        OutData = USERINPUT_SegMethod["func"](
            USERINPUT_Image, 
            **USERINPUT_SegMethod["params"],
            visualise=True
        )
        SEG_MAP = OutData["map"]

        # Display Outputs
        st.markdown("## Segmentation Output")
        st.pyplot(ImageVis_SegmentationMap(
            USERINPUT_Image, SEG_MAP,
            classes=SegmentationClasses_Default(SEG_MAP.shape[2]),
            cmap=DEFAULT_CMAP
        ))
        # Display Visualisations
        UI_DisplayVisData(OutData)

def video_feed_segmentation():
    # Title
    st.markdown("# Segmentation - Video Feed")

    # Load Inputs
    # Video
    USERINPUT_VideoMode = st.selectbox("Video Mode", ["Webcam", "URL"])
    if USERINPUT_VideoMode == "Webcam":
        USERINPUT_VideoFeedObj = VideoUtils_ReadWebcam()
    else:
        USERINPUT_VideoURL = st.text_input("Youtube URL", "https://www.youtube.com/watch?v=KsH2LA8pCjo")
        USERINPUT_VideoFeedObj = VideoUtils_ReadVideoURL_Youtube(USERINPUT_VideoURL)
    # Method
    USERINPUT_SegMethod = UI_LoadSegAlgo()
    USERINPUT_Visualise = st.checkbox("Visualisations", value=False)

    # Process Inputs
    USERINPUT_Process = st.checkbox("Stream Process", value=False)
    if not USERINPUT_Process: USERINPUT_Process = st.button("Run Segmentation")
    if USERINPUT_Process:
        # Init
        USERINPUT_VideoFeed = VideoUtils_GetVideoLoader(
            USERINPUT_VideoFeedObj,
            max_frames=-1, skip_faulty_frames=True
        )
        # Init UI
        UI_Elements = {
            "input": {
                "title": st.empty(),
                "image": st.empty()
            },
            "output": {
                "title": st.empty(),
                "output": st.empty()
            },
            "vis": {
                "title_1": st.empty(),
                "title_2": st.empty(),
                "plots": {}
            }
        }
        # Loop through Video Feed
        while True:
            # Read Frame
            frameCount, USERINPUT_Image = next(USERINPUT_VideoFeed)
            # Fix Frame
            USERINPUT_Image = cv2.cvtColor(USERINPUT_Image, cv2.COLOR_BGR2RGB)
            USERINPUT_Image = np.array(USERINPUT_Image, dtype=float) / 255.0
            # Segementation
            OutData = USERINPUT_SegMethod["func"](
                USERINPUT_Image, 
                **USERINPUT_SegMethod["params"],
                visualise=USERINPUT_Visualise
            )
            SEG_MAP = OutData["map"]
            # Display Inputs
            UI_Elements["input"]["title"].markdown("## Input")
            UI_Elements["input"]["image"].image(USERINPUT_Image)
            # Display Outputs
            UI_Elements["output"]["title"].markdown("## Segmentation Output")
            UI_Elements["output"]["output"].pyplot(ImageVis_SegmentationMap(
                USERINPUT_Image, SEG_MAP,
                classes=SegmentationClasses_Default(SEG_MAP.shape[2]),
                cmap=DEFAULT_CMAP
            ))
            # Display Visualisations
            if USERINPUT_Visualise:
                UI_Elements["vis"]["title_1"].markdown("# Visualisations")
                UI_Elements["vis"]["title_1"].markdown("## Plots")
                # Plots
                for k in OutData["figs"]["pyplot"].keys():
                    if k not in UI_Elements["vis"]["plots"].keys():
                        UI_Elements["vis"]["plots"][k] = {
                            "title": st.empty(),
                            "cols": [col.empty() for col in st.columns(len(OutData["figs"]["pyplot"][k]))]
                        }
                    UI_Elements["vis"]["plots"][k]["title"].markdown(f"### {k}")
                    for i in range(len(OutData["figs"]["pyplot"][k])):
                        UI_Elements["vis"]["plots"][k]["cols"][i].pyplot(OutData["figs"]["pyplot"][k][i])

def dataset_segmentation_run():
    # Title
    st.markdown("# Segmentation - Dataset Run")

    # Load Inputs
    # Dataset
    USERINPUT_Dataset = UI_LoadDataset()
    # Method
    USERINPUT_SegMethod = UI_LoadSegAlgo()
    # Other
    cols = st.columns(2)
    USERINPUT_SaveMaps = cols[0].checkbox("Save Generated Maps")
    USERINPUT_DisplayLatestProcessedSample = cols[1].checkbox("Display Latest Processed Sample", value=True)
    # Process Inputs
    if st.button("Run Dataset Segmentation"):
        TestData = {
            "y_pred": [],
            "time_exec": [],
            "image_size": {
                "height": [],
                "width": []
            }
        }
        PROGRESS_OBJ = st.progress(0.0)
        CURRENT_RESULT_OBJS = {
            "image": st.empty(),
            "pred": {
                "title": st.empty(),
                "map": st.empty()
            }
        }
        for i in range(USERINPUT_Dataset.shape[0]):
            I_path = USERINPUT_Dataset["path"][i]
            # Read Image
            I_bytes = open(I_path, "rb").read()
            I = ImageUtils_Bytes2Array(I_bytes)
            TestData["image_size"]["height"].append(I.shape[0])
            TestData["image_size"]["width"].append(I.shape[1])
            # Segmenation
            START_TIME = time.time()
            MAP_PRED = USERINPUT_SegMethod["func"](
                I, 
                **USERINPUT_SegMethod["params"],
                visualise=False
            )["map"]
            END_TIME = time.time()
            # Update Progress
            TestData["y_pred"].append(MAP_PRED)
            TestData["time_exec"].append(END_TIME - START_TIME)
            PROGRESS_OBJ.progress((i+1) / USERINPUT_Dataset.shape[0])
            if USERINPUT_DisplayLatestProcessedSample:
                CURRENT_RESULT_OBJS["image"].image(I_path, caption=f"Image: {i}", use_column_width=True)
                CURRENT_RESULT_OBJS["pred"]["title"].markdown("Predicted:")
                CURRENT_RESULT_OBJS["pred"]["map"].pyplot(ImageVis_SegmentationMap(
                    I, MAP_PRED,
                    classes=SegmentationClasses_Default(MAP_PRED.shape[2]),
                    cmap=DEFAULT_CMAP
                ))
            # Save Map
            if USERINPUT_SaveMaps: ImageUtils_SaveMap(MAP_PRED, USERINPUT_Dataset["map_path"][i])
        # Display Outputs
        # Test Times
        st.markdown("## Test Times")
        Times = {
            "Min Time": float(np.min(TestData["time_exec"])),
            "Max Time": float(np.max(TestData["time_exec"])),
            "Mean Time": float(np.mean(TestData["time_exec"])),
            "Median Time": float(np.median(TestData["time_exec"])),
            "Std Time": float(np.std(TestData["time_exec"]))
        }
        st.write(Times)

def dataset_segmentation_test():
    # Title
    st.markdown("# Segmentation - Dataset Test")

    # Load Inputs
    # Dataset
    USERINPUT_Dataset = UI_LoadDataset()
    # Method
    USERINPUT_SegMethod = UI_LoadSegAlgo()

    USERINPUT_DisplayLatestProcessedSample = st.checkbox("Display Latest Processed Sample", value=True)
    # Process Inputs
    if st.button("Run Dataset Test"):
        TestData = {
            "y_true": [],
            "y_pred": [],
            "time_exec": [],
            "image_size": {
                "height": [],
                "width": []
            }
        }
        PROGRESS_OBJ = st.progress(0.0)
        CURRENT_RESULT_OBJS = {
            "image": st.empty(),
            "true": {
                "title": st.empty(),
                "map": st.empty()
            },
            "pred": {
                "title": st.empty(),
                "map": st.empty()
            }
        }
        for i in range(USERINPUT_Dataset.shape[0]):
            I_path = USERINPUT_Dataset["path"][i]
            MAP_TRUE = ImageUtils_LoadMap(USERINPUT_Dataset["map_path"][i])
            # Read Image
            I_bytes = open(I_path, "rb").read()
            I = ImageUtils_Bytes2Array(I_bytes)
            TestData["image_size"]["height"].append(I.shape[0])
            TestData["image_size"]["width"].append(I.shape[1])
            # Segmenation
            START_TIME = time.time()
            MAP_PRED = USERINPUT_SegMethod["func"](
                I, 
                **USERINPUT_SegMethod["params"],
                visualise=False
            )["map"]
            END_TIME = time.time()
            # Update Progress
            TestData["y_true"].append(MAP_TRUE)
            TestData["y_pred"].append(MAP_PRED)
            TestData["time_exec"].append(END_TIME - START_TIME)
            PROGRESS_OBJ.progress((i+1) / USERINPUT_Dataset.shape[0])
            if USERINPUT_DisplayLatestProcessedSample:
                CURRENT_RESULT_OBJS["image"].image(I_path, caption=f"Image: {i}", use_column_width=True)
                CURRENT_RESULT_OBJS["true"]["title"].markdown("True:")
                CURRENT_RESULT_OBJS["true"]["map"].pyplot(ImageVis_SegmentationMap(
                    I, MAP_TRUE,
                    classes=SegmentationClasses_Default(MAP_TRUE.shape[2]),
                    cmap=DEFAULT_CMAP
                ))
                CURRENT_RESULT_OBJS["pred"]["title"].markdown("Predicted:")
                CURRENT_RESULT_OBJS["pred"]["map"].pyplot(ImageVis_SegmentationMap(
                    I, MAP_PRED,
                    classes=SegmentationClasses_Default(MAP_PRED.shape[2]),
                    cmap=DEFAULT_CMAP
                ))
        # Display Outputs
        UI_DisplayTestResults(TestData)

def image_process():
    # Title
    st.markdown("# Image Process")

    # Image
    USERINPUT_Image = UI_LoadImage()
    # Clean and Edit Image
    USERINPUT_Image = UI_ImageEdit(USERINPUT_Image)
    
#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()