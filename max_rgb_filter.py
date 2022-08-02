# https://pyimagesearch.com/2015/09/28/implementing-the-max-rgb-filter-in-opencv/
import cv2
import numpy as np


def filtering(img0: np.ndarray):
    b, g, r = cv2.split(img0)

    m = np.maximum(r, np.maximum(g, b))

    r[r < m] = 0
    g[g < m] = 0
    b[b < m] = 0

    return cv2.merge([b, g, r])