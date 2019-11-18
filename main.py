from common import recv_image
from bg_subtraction import bg_subtraction
from my_image_subscriber import socket_setup
import numpy as np
import cv2 as cv
from utilities import *
from constants import *

def main():
    socket = socket_setup()
    min_area = (FRAME_H * FRAME_W)*0.04

    while True:
        # print("Try to get background")
        # meta_data, image = recv_image(socket)
        # img = np.array(image)[:,:,::-1]
        # if img is None:
        #     continue
        try:
            print("Try to get background")
            meta_data, image = recv_image(socket)
            img = np.array(image)[:, :, ::-1]
            if img is None:
                continue
        except:
            continue

        bg = img.copy()
        cv.imshow("background", bg)
        k = cv.waitKey(-1) & 0xff
        if k == ord('q'):
            break
        else:
            continue
    bg = cv.cvtColor(bg.copy(), cv.COLOR_BGR2GRAY)
    print("Get bg success")
    while True:
        try:
            meta_data, image = recv_image(socket)
            img = np.array(image)[:, :, ::-1]
            if img is None:
                continue

        except:
            continue
        # fg = backSub.apply(frame)
        obj = bg_subtraction(img, bg.copy(), mode='neg')
        person = cv.bitwise_and(img, img, mask=obj)

        _,th = cv.threshold(obj, 127, 255, cv.THRESH_BINARY)
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

        cv.imshow("obj", obj)
        # cv.imshow("img", img)
        # cv.imshow("bg", bg)
        cv.imshow("person", person)
        cv.imshow("result", result)
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break


if __name__ == "__main__":
    main()
