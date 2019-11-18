from common import recv_image
from bg_subtraction import bg_subtraction_bgr
from my_image_subscriber import socket_setup
import numpy as np
import cv2 as cv
from constants import *
from utilities import *
from keras.models import load_model
from submission import submission
import argparse
import time

model_file = MODEL_PATH + r"/model-118-0.0052.hdf5"
model = load_model(model_file)
reponse_payload = {
    "scene_no": SCENE_NO,
    "ppe": {
        "helmet": False,
        "coverall": False,
        "boots": False,
        "gloves": False
    }
}


def check_is_object(binary):
    print("Check is object")
    r, c = binary.shape
    white = np.count_nonzero(binary)
    ratio = white/(r*c)
    print("ratio", ratio)
    return ratio >= 0.15


def object_detection(img):
    print("Object Detection")
    height, width, _ = img.shape
    print("H W:", height, width)
    lower_bound = {
        'helmet': 0,
        'coverall': int(height*0.15),
        'boots': int(height*0.55),
        'gloves': int(height*0.25),
    }
    upper_bound = {
        'helmet': int(height*0.25),
        'coverall': int(height*0.8),
        'boots': int(height),
        'gloves': int(height*0.7),
    }
    bounding = img.copy()
    result = img.copy()
    for k in lower_bound.keys():
        print("KEY:", k)
        lower = lower_bound[k]
        upper = upper_bound[k]
        cv.rectangle(bounding,(0,lower),(width,upper),(0,0,255),2)

        print("ROI", lower, upper)
        roi = img.copy()[lower:upper, :]

        print("ROI Shape", roi.shape)
        hsv = cv.cvtColor(roi.copy(), cv.COLOR_BGR2HSV)
        
        cv.imshow("HSV "+k, hsv)

        color_th = cv.inRange(
            hsv.copy(), COLOR_SEGMENT[k]['lower'], COLOR_SEGMENT[k]['upper'])
        color_th = cv.erode(color_th, get_kernel())

        _, contours, _ = cv.findContours(
            color_th, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        result = img.copy()
        cv.drawContours(result, contours, -1, (255, 255, 0), 2)

        for cnt in contours:
            area = cv.contourArea(cnt)
            rect = cv.minAreaRect(cnt)
            (x,y),(w,h),angle = rect
            print("AREA",MIN_AREA[k],area,MAX_AREA[k])
            if area < MIN_AREA[k]:
                continue
            if area > MAX_AREA[k]:
                continue
            if k == 'gloves':
                if max(w,h)/min(w,h) >= 1.25:
                    continue
            if k == 'boots':
                if max(w,h)/min(w,h) >= 1.5:
                    continue
            update_payload(k)
            cv.waitKey(1)
            break
        # cv.imshow("Object detection Result",result)
        cv.imshow("Th "+k, color_th)
        cv.waitKey(1)
    cv.imshow("bounding",bounding)       

def predict(image):
    global model
    print("prediction")
    rows, cols, ch = image.shape
    frame = image.copy()
    frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (256, 256))
    frame = frame.reshape((1, 256, 256, 3))
    frame = frame.astype('float32')
    frame = (frame / 255.)
    pred = model.predict(frame)[0]

    pred = cv.resize(pred.copy(), (cols, rows))
    pred = cv.cvtColor(pred.copy(), cv.COLOR_RGB2BGR)
    pred = pred * 255.
    pred = pred.astype('uint8')
    cv.imshow("Prediction", pred)
    return pred


def update_payload(name):
    global reponse_payload
    print("Update Payload")
    reponse_payload['ppe'][name] = True
    print(name, reponse_payload['ppe'][name])
    status = np.zeros((260,260,3),np.uint8)
    x = 10
    y = 40
    for k in COLOR_SEGMENT.keys():
        text = k
        if reponse_payload['ppe'][k]:
            cv.putText(status, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1,cv.LINE_AA) 
        else:
            cv.putText(status, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1,cv.LINE_AA) 
            
        y += 40
    cv.imshow("PPE STATUS",status)

def main(vdo_path):
    global reponse_payload

    cap = cv.VideoCapture(vdo_path)

    start_time = time.time()

    min_area = (FRAME_H * FRAME_W)*0.03
    get_bg = False
    x = 40
    for k in COLOR_SEGMENT:
        win_name = "Th "+k
        cv.namedWindow(win_name)
        cv.resizeWindow(win_name,300,300)
        cv.moveWindow(win_name,x,0)
        x += 300
    x = 40
    
    for k in COLOR_SEGMENT:
        win_name = "HSV "+k
        cv.namedWindow(win_name)
        cv.resizeWindow(win_name,300,300)
        cv.moveWindow(win_name,x,300)
        x += 300

    status = np.zeros((260,260,3),np.uint8)
    x = 10
    y = 40
    for k in COLOR_SEGMENT.keys():
        text = k
        cv.putText(status, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1,cv.LINE_AA)     
        y += 40

    cv.imshow("PPE STATUS",status)

    out = cv.VideoWriter('./out1.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (570,480))

    while True:
        print("In loop")
        ret, image = cap.read()

        try:
            if image is None:
                print("Image is None")
                break
            img = image[:, 50:-20]
        except:
            print("Error exception")
            continue
        
        if not get_bg:
            print("Get Background")
            for i in range(5):
                _, img = cap.read()
                img = img[:, 50:-20]

            bg = img.copy()
            bg = bg.copy()
            bg = cv.medianBlur(bg, 3)
            cv.imwrite("./first.png",bg)
            get_bg = True
            # cv.imshow("gray", bg)
            for i in range(5):
                _, _ = cap.read()
            continue

        obj = bg_subtraction_bgr(img, bg.copy())
        person = cv.bitwise_and(img, img, mask=obj)

        _, th = cv.threshold(obj, 127, 255, cv.THRESH_BINARY)
        th = cv.dilate(th, get_kernel(ksize=(3, 7)))
        th = cv.erode(th, get_kernel(),iterations=1)
        _, contours, _ = cv.findContours(
            th, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        result = img.copy()
        cv.drawContours(result, contours, -1, (255, 0, 0), 2)

        # cv.imshow("th", th)

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_area:
                continue
            (x, y, w, h) = cv.boundingRect(cnt)
            if w > FRAME_W * 0.4:
                print("Continues...")
                print("Width of object is too large")
                continue
            if h < FRAME_H*0.3:
                print("Continues...")
                print("Height of object is too short")
                continue

            # add_h = h*0.25
            add_h = 0

            y = max(0, y-add_h)
            h = min(FRAME_H, h+add_h)
            y = int(y)
            h = int(h)
            cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = img[y:y+h, x:x+w]
            predicted = predict(roi)
            object_detection(predicted)

        print(reponse_payload)
        out.write(img)
        cv.imshow("obj", th)
        cv.imshow("person", person)
        cv.imshow("result", result)
        k = cv.waitKey(50) & 0xff

        if k == ord('q'):
            out.release()
            break


if __name__ == "__main__":
    # vdo = VDO_PATH + "/long_video.avi"
    vdo = VDO_PATH + "/tul_all.avi"
    # vdo = VDO_PATH + "/long_video_03.avi"
    # vdo = VDO_PATH + "/tul_hand_fake.avi"
    
    main(vdo)
