import cv2 as cv
from utilities import *
from constants import *
import numpy as np
import time 

def rename():
    dir = IMG_PATH + r"/dataset_ice"
    file_list = get_file_path(dir)
    r_list = np.array([])
    c_list = np.array([])
    for f in file_list:
        img = cv.imread(f)
        r,c,_ = img.shape
        r_list = np.append(r_list,r)
        c_list = np.append(c_list,c)
        name = time.time()
        name = str(name).replace(".","")
        name += '.jpg'
        cv.imwrite(IMG_PATH + r'/' + name, img)
    print(r_list.mean())
    print(c_list.mean())

def resize():
    dir = IMG_PATH + r"/dataset_use"
    file_list = get_file_path(dir)
    r_list = np.array([])
    c_list = np.array([])
    for f in file_list:
        img = cv.imread(f)
        img = cv.resize(img,(128,256))
        r,c,_ = img.shape
        r_list = np.append(r_list,r)
        c_list = np.append(c_list,c)
        name = time.time()
        name = str(name).replace(".","")
        name += '.jpg'
        cv.imwrite(IMG_PATH + r'/dataset_resize/' + name, img)
    print(r_list.mean())
    print(c_list.mean())

# rename()
resize()