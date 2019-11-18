from common import recv_image
from bg_subtraction import bg_subtraction
from my_image_subscriber import socket_setup
import numpy as np
import cv2 as cv
from constants import *
from utilities import *
import time

def capture():
    name = "out.avi"
    cap = cv.VideoCapture(VDO_PATH + "/" + name)
    while True:
        ret, img = cap.read()
        if ret:
            cv.imwrite(IMG_PATH + "/bg.jpg", img)
            break
def main():
    
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
            if h < FRAME_H*0.5:
                continue
            if h/w < 1.5:
                continue 
            # wh_ratio = 1.*w/h
            add_h = h*0.25
            y = max(0,y-add_h)
            h = min(FRAME_H, h+add_h)
            y = int(y)
            h = int(h)
            cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)  
            roi = img[y:y+h,x:x+w]
            # print(PERSON_PATH + "/" + str(time).split(".")[0] + ".jpg")
            cv.imshow("result", result)
            if cv.waitKey(-1) & 0xff == ord('s'):
                cv.imwrite(PERSON_PATH + "/" + str(time.time()).split(".")[0] + ".jpg",roi.copy())
            
        # roi =None
        frame_count += 1
        cv.imshow("obj", obj)
        cv.imshow("person", person)
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
    cap.release()

if __name__ == "__main__":
    main()