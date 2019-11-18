import cv2 as cv
import numpy as np
import os


def get_kernel(shape='rect', ksize=(3, 3)):
    if shape == 'rect':
        return cv.getStructuringElement(cv.MORPH_RECT, ksize)
    elif shape == 'ellipse':
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
    elif shape == 'plus':
        return cv.getStructuringElement(cv.MORPH_CROSS, ksize)
    else:
        return None


def get_file_path(dir_name):
    """
      Get all files in directory "dir_name"
    """
    file_list = os.listdir(dir_name)
    files = []
    for f in file_list:
        abs_path = os.path.join(dir_name, f)
        if os.path.isdir(abs_path):
            files = files + get_file_path(abs_path)
        else:
            files.append(abs_path)
    return files


def get_file_name(img_path):
    if "\\" in img_path:
        name = img_path.split('\\')[-1]
    else:
        name = img_path.split('/')[-1]

    name = name.replace('.gif', '')
    name = name.replace('.png', '')
    name = name.replace('.jpg', '')
    return name
