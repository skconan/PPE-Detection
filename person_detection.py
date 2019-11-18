from common import recv_image
from bg_subtraction import bg_subtraction
from my_image_subscriber import socket_setup
import numpy as np
import cv2 as cv
from constants import *
from utilities import *
from keras.models import load_model

model_file = MODEL_PATH + r"/model-color-obj-bg.hdf5"


def color_detection(img):
    lower = [160,0,0]
    upper = [179, 255,255]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array(lower,np.uint8)
    upper = np.array(upper,np.uint8)

def capture():
    name = "out.avi"
    cap = cv.VideoCapture(VDO_PATH + "/" + name)
    while True:
        ret, img = cap.read()
        if ret:
            cv.imwrite(IMG_PATH + "/bg.jpg", img)
            break
def main():
    vdo_path = r"D:\hackathon\hackathon_image_streaming\videos"
    # bg = cv.imread(IMG_PATH + "/bg.jpg",0)
    # bg = cv.GaussianBlur(bg, (11, 11), 0)
    name = "out_all_forward.avi"
    name = "out_all_no_hand.avi"
    # vdo_name = 
    # cap = cv.VideoCapture()
    cap = cv.VideoCapture(VDO_PATH + "/" + name)
    min_area = (FRAME_H * FRAME_W)*0.04
    # back_sub = cv.createBackgroundSubtractorKNN()
    get_bg = False
    frame_count = 0
    while True:
        try:
            ret, img = cap.read()
            if not ret:
                print("Image is None")
                break
        except:
            continue
        if not get_bg:
            bg = img.copy()
            bg = cv.cvtColor(bg,cv.COLOR_BGR2GRAY)
            get_bg = True
        # fg = backSub.apply(frame)
        obj = bg_subtraction(img, bg.copy(), mode='neg')
        person = cv.bitwise_and(img, img, mask=obj)

        _,th = cv.threshold(obj, 127, 255, cv.THRESH_BINARY)
        # th = cv.erode(th,get_kernel(ksize=(5,5)))
        th = cv.dilate(th,get_kernel())
        cv.imshow("th",th)
        _,contours,_ = cv.findContours(th,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        frame_move = 0
        result = img.copy()
        cv.drawContours(result,contours,-1,(255,0,0),2)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_area:
                continue
            (x, y, w, h) = cv.boundingRect(cnt)
            if w > FRAME_W * 0.3:
                continue
            # wh_ratio = 1.*w/h
            add_h = h*0.2
            y = max(0,y-add_h)
            h = min(FRAME_H, h+add_h)
            y = int(y)
            h = int(h)
            cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    
        frame_count += 1
        # if frame_count >= 30:
        #     frame_count = 0
        #     get_bg = False
        cv.imshow("obj", obj)
        # cv.imshow("img", img)
        # cv.imshow("bg", bg)
        cv.imshow("person", person)
        cv.imshow("result", result)
        k = cv.waitKey(100) & 0xff
        if k == ord('q'):
            break
    cap.release()

if __name__ == "__main__":
    main()