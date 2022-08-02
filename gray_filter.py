import cv2


def gray_imp(img0):
    b, g, r = cv2.split(img0)
    return r * 0.21 + g * 0.72 + b * 0.07
