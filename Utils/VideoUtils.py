"""
Video Utils
"""

# Imports
import pafy
import requests

from .ImageUtils import *

# Main Classes
class VideoInput:
    '''
    Psuedo Video Input for URL
    '''
    def __init__(self, url):
        self.url = url

    def isOpened(self):
        return True

    def release(self):
        pass

    def read(self):
        I = None
        try:
            I_response = requests.get(self.url).content
            I_arr = np.array(bytearray(I_response), dtype=np.uint8)
            I = cv2.imdecode(I_arr, -1)
        except:
            pass
        if I is None:
            I = np.zeros((480, 640, 3), np.uint8)
            print("Video Input Error")
        return True, I

class VideoInput_Youtube:
    '''
    Psuedo Video Input for Youtube URLs
    '''
    def __init__(self, url):
        vidObj = pafy.new(url)
        urlObj = vidObj.getbest(preftype="any")
        self.urlObj = urlObj
        self.url = urlObj.url
        self.vid = cv2.VideoCapture(urlObj.url)

    def isOpened(self):
        return True

    def release(self):
        pass

    def read(self):
        I = None
        try:
            check, I = self.vid.read()
        except:
            pass
        if I is None:
            I = np.zeros((480, 640, 3), np.uint8)
            print("Video Input Error")
        return True, I

# Main Functions
# Basic Functions
def VideoUtils_ReadVideo(path):
    '''
    Read Video
    '''
    return cv2.VideoCapture(path)

def VideoUtils_ReadWebcam():
    '''
    Read Webcam
    '''
    return cv2.VideoCapture(0)

def VideoUtils_ReadVideoURL(url):
    '''
    Read Video from URL
    '''
    return VideoInput(url)

def VideoUtils_ReadVideoURL_Youtube(url):
    '''
    Read Video from Youtube URL
    '''
    return VideoInput_Youtube(url)

# Video Loader Functions
def VideoUtils_VideoLoader(
    vid, 
    
    max_frames=-1, 
    skip_faulty_frames=False,
    
    **params
    ):
    '''
    Video Loader - Simple
    '''
    # Init
    frameCount = 0
    # Loop and read frames
    while(vid.isOpened() and ((not (frameCount == max_frames)) or (max_frames == -1))):
        # Read frame
        ret, frame = vid.read()
        # Update frameCount if successful else frame is None
        if ret: frameCount += 1
        else: frame = None
        # Skip Faulty Frames Check
        if skip_faulty_frames and frame is None: continue
        # Return frame
        yield frameCount, frame
    # Cleanup
    vid.release()

# Load Functions
def VideoUtils_GetVideoLoader(
    vid,

    max_frames=-1, 
    skip_faulty_frames=False,

    **params
    ):
    '''
    Get Video Loader - Simple
    '''
    # Check if camera opened successfully
    if not vid.isOpened(): print("Error opening video stream or file")
    # Form Video Loader
    loader = VideoUtils_VideoLoader(vid, max_frames)

    return loader

def VideoUtils_SaveFrames2Video(frames, pathOut, fps=20, size=None):
    '''
    VideoUtils - Save Frames to Video
    '''
    if os.path.splitext(pathOut)[-1] == ".gif":
        
        frames_images = [Image.fromarray(np.array(frame*255, dtype=np.uint8)) for frame in frames]
        extraFrames = []
        if len(frames_images) > 1: extraFrames = frames_images[1:]
        frames_images[0].save(pathOut, save_all=True, append_images=extraFrames, format="GIF", loop=0)
    else:
        # if size is None: size = (640, 480)
        # frames = [np.array(frame*255, dtype=int) for frame in frames]
        # frames = [ResizeImage_Pad(frame, size=size[::-1]) for frame in frames]
        if size is None: size = (frames[0].shape[1], frames[0].shape[0])
        frames = [np.array(frame*255, dtype=np.uint8) for frame in frames]
        codec = cv2.VideoWriter_fourcc(*'AVC1')
        out = cv2.VideoWriter(pathOut, codec, fps, size)
        for frame in frames:
            out.write(frame)
        # out.close()
        out.release()