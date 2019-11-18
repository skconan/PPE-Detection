import numpy as np
WS_PATH = r"E:\hackathon"
VDO_PATH = WS_PATH + r"\videos"
MODEL_PATH = WS_PATH + r"\models"
IMG_PATH = WS_PATH + r"\images"
PERSON_PATH = WS_PATH + r"\person"
FRAME_W = 640
FRAME_H = 480
TIMEOUT = 50
COLOR_SEGMENT = {
    'helmet': {
        'upper': np.array([38, 255, 255], np.uint8), 'lower': np.array([20, 220, 0], np.uint8)
    },
    'gloves':
    {
        'upper': np.array([80, 255, 255], np.uint8), 'lower': np.array([50, 220, 0], np.uint8)
    },
    'boots': {
        'upper': np.array([25, 255, 255], np.uint8), 'lower': np.array([0, 220, 0], np.uint8)

    },
    'coverall': {
        'upper': np.array([155, 255, 255], np.uint8), 'lower': np.array([135, 220, 0], np.uint8)
    }
}
MIN_AREA = {
    'helmet': 1000,
    'gloves': 100,
    'boots': 500,
    'coverall': 8000, 
}
MAX_AREA = {
    'helmet': 2000,
    'gloves': 150,
    'boots': 800,
    'coverall': 20000, 
}
IS_TEST = False
SCENE_NO = 10