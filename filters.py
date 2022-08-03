import numpy as np
from scipy import signal
import cv2


def contrast(img: np.ndarray, value: int = 30):
    img_filter = img.copy()
    img_filter[img_filter < value] = 0
    img_filter[(img_filter >= value) & (img_filter <= 128)] -= value
    img_filter[img_filter > 255 - value] = 255
    img_filter[(img_filter < 255 - value) & (img > 128)] += value
    return img_filter


def laplacian(img: np.ndarray, kernel: np.ndarray = np.array([[0, 1, 0],
                                                              [1, -4, 1],
                                                              [0, 1, 0]]), scale: int = 1):
    img_filter = gray_imp(img)
    return signal.convolve2d(img_filter, kernel)


def gaussian_blur(img0: np.ndarray, kernel: np.ndarray = np.array([[0.0625, 0.125, 0.0625],
                                                                   [0.125, 0.25, 0.125],
                                                                   [0.0625, 0.125, 0.0625]])):
    """ Remove noise """
    r, g, b = cv2.split(img0)
    r = signal.convolve2d(r, kernel)
    r = np.pad(r, 2)
    g = signal.convolve2d(g, kernel)
    g = np.pad(g, 2)
    b = signal.convolve2d(b, kernel)
    b = np.pad(b, 2)
    return cv2.merge([r, g, b])


def gray_imp(img0):
    b, g, r = cv2.split(img0)
    return r * 0.21 + g * 0.72 + b * 0.07


def filtering(img0: np.ndarray):
    b, g, r = cv2.split(img0)

    m = np.maximum(r, np.maximum(g, b))

    r[r < m] = 0
    g[g < m] = 0
    b[b < m] = 0

    return cv2.merge([b, g, r])
