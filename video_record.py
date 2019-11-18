import sys
import zmq
from common import recv_image


import numpy as np
import argparse
import time
import cv2 as cv
from my_image_subscriber import socket_setup
def main(args):
    # Socket to talk to server
    socket = socket_setup()
    last_time = time.time()
    print("Start")
    # try:
        # out = cv.VideoWriter('out.avi',cv.VideoWriter_fourcc('M','J','P','G'), 60, (640,480))
        # while True:
            # meta_data, image = recv_image(socket)
            # print(meta_data)
            # if args.draw:
                # img = np.array(image)[:,:,::-1]
                # out.write(img)
                # cv.imshow('image', img)
                # k = cv.waitKey(1) & 0xff
            # 
            # if k==ord('q'):
                # out.release()
                # break
# 
    # except KeyboardInterrupt:
        # print("Closed subscribe")
    out = cv.VideoWriter('out_all_no_helmet_extra_forward.avi',cv.VideoWriter_fourcc('M','J','P','G'), 60, (640,480))
    while True:
        meta_data, image = recv_image(socket)
        print(meta_data)
        img = np.array(image)[:,:,::-1]
        out.write(img)
        cv.imshow('image', img)
        k = cv.waitKey(1) & 0xff
        
        if k==ord('q'):
            out.release()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--port', default="5560", help='port')
    parser.add_argument('--host', default="localhost", help='host')
    parser.add_argument('--draw', dest='draw', action='store_true', help='Draw images (cv2)')
    parser.add_argument('--no-draw', dest='draw', action='store_false')
    parser.set_defaults(draw=False)
    args = parser.parse_args()
    main(args)
