import cv2 as cv
import numpy as np


def normalize(gray):
    a = (gray.max()-gray.min()) + 1
    return np.uint8(255*(gray-gray.min())/a)


def bg_subtraction_bgr(img,bg):
    sub_bgr = np.int16(cv.medianBlur(img.copy(), 3)) - \
        np.int16(bg.copy())
    sub_neg = np.clip(sub_bgr.copy(), sub_bgr.copy().min(), 0)
    sub_neg = np.uint8(np.abs(sub_neg))

    sub_pos = np.clip(sub_bgr.copy(), 0, 255)
    sub_pos = np.uint8(sub_pos)
    res = cv.bitwise_or(sub_pos, sub_neg)
    res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    cv.imshow("res_gray",res_gray)
    thval, th_new = cv.threshold(
        res_gray, 20, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    _, result = cv.threshold(
        res_gray, thval*0.5, 255, cv.THRESH_BINARY
    )
    return result

def bg_subtraction(fg, bg, bg_k=1, fg_k=3, mode='neg'):
    # hsv = cv.cvtColor(fg, cv.COLOR_BGR2HSV)
    # _,fg,_ = cv.split(hsv)
    fg = cv.cvtColor(fg.copy(), cv.COLOR_BGR2GRAY)
    fg = cv.medianBlur(fg, 3)

    # start_time = rospy.Time.now()
    # bg = cv.medianBlur(bg.copy(), 15)
    # fg = cv.medianBlur(fg.copy(), 7)
    # fg = kmean(gray, k=fg_k)

    sub_sign = np.int16(fg) - np.int16(bg)
    # sub_sign = sub_sign**2
    # sub_sign[sub_sign < sub_sign.mean()] = 0
    sub_neg = np.clip(sub_sign.copy(), sub_sign.copy().min(), 0)
    sub_neg = np.uint8(np.abs(sub_neg))

    sub_pos = np.clip(sub_sign.copy(), 0, 255)
    sub_pos = np.uint8(sub_pos)
    # sub_pos = normalize(sub_pos)

    cv.imshow("sub_neg", sub_neg)
    cv.imshow("sub_pos", sub_pos)

    # sub_sign_neg = sub_sign.copy()
    # sub_sign_pos = sub_sign.copy()

    # sub_sign_neg[sub_sign_neg < 127 - 30] = 0
    # sub_sign_pos[sub_sign_pos > 127 + 30] = 0

    # cv.imshow("sub_sign_neg", sub_sign_neg)
    # cv.imshow("sub_sign_pos", sub_sign_pos)

    th1, result = cv.threshold(
        sub_neg, 20, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    th2, result1 = cv.threshold(
        sub_pos, 20, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    const = 0.5
    _, result = cv.threshold(
        sub_neg, th1*0.5, 255, cv.THRESH_BINARY
    )

    _, result1 = cv.threshold(
        sub_pos, th2*const, 255, cv.THRESH_BINARY
    )

    cv.imshow("sub_sign", normalize(sub_sign))

    result = cv.bitwise_or(result, result1)
    # cv.imshow("sub_sign", normalize())
    cv.imshow('fg', fg)
    cv.imshow('bg', bg)

    if mode == 'neg':
        sub_neg = np.clip(sub_sign.copy(), sub_sign.copy().min(), 0)
        sub_neg = sub_neg**2
        sub_neg = normalize(sub_neg)
        cv.imshow('sub_neg', sub_neg)
        _, result = cv.threshold(
            sub_neg, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )
        cv.imshow("sub_neg", sub_neg.copy())
        # _, result = cv.threshold(
        #     sub_neg, 127, 255, cv.THRESH_BINARY
        # )
    elif mode == 'pos':
        sub_pos = np.clip(sub_sign.copy(), 0, sub_sign.copy().max())
        sub_pos = normalize(sub_pos)
        cv.imshow('sub_pos', sub_pos)

        _, result = cv.threshold(
            sub_pos, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

    else:
        sub_sign = np.int16(fg) - np.int16(bg)
        # sub_sign = sub_sign**2
        # sub_sign[sub_sign < sub_sign.mean()] = 0
        sub_neg = np.clip(sub_sign.copy(), sub_sign.copy().min(), 0)
        sub_neg = np.uint8(np.abs(sub_neg))

        sub_pos = np.clip(sub_sign.copy(), 0, 255)
        sub_pos = np.uint8(sub_pos)
        # sub_pos = normalize(sub_pos)

        cv.imshow("sub_neg", sub_neg)
        cv.imshow("sub_pos", sub_pos)

        # sub_sign_neg = sub_sign.copy()
        # sub_sign_pos = sub_sign.copy()

        # sub_sign_neg[sub_sign_neg < 127 - 30] = 0
        # sub_sign_pos[sub_sign_pos > 127 + 30] = 0

        # cv.imshow("sub_sign_neg", sub_sign_neg)
        # cv.imshow("sub_sign_pos", sub_sign_pos)

        th1, result = cv.threshold(
            sub_neg, 20, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )

        th2, result1 = cv.threshold(
            sub_pos, 20, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )
        const = 0.5
        _, result = cv.threshold(
            sub_neg, th1*0.5, 255, cv.THRESH_BINARY
        )

        _, result1 = cv.threshold(
            sub_pos, th2*const, 255, cv.THRESH_BINARY
        )

        cv.imshow("sub_sign", normalize(sub_sign))

        result = cv.bitwise_or(result, result1)
    # cv.imshow("sub_neg",sub.copy())
    # time_duration = rospy.Time.now()-start_time
    # print(time_duration.to_sec())
    # cv.imshow("fg",fg)
    # cv.imshow("bg",bg)

    return result
