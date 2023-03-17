import numpy as np
from scipy import signal
import cv2

# All number values is taken from the OpenCv documentation
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
    # """Laplacian edge detection"""
    # cols = (img.shape[0] - kernel.shape[0]) // scale + 1
    # rows = (img.shape[1] - kernel.shape[0]) // scale + 1
    #
    # lst = np.empty((cols, rows), dtype="int32")
    #
    # for i in range(0, cols * scale, scale):
    #     for j in range(0, rows * scale, scale):
    #         lst[i // scale, j // scale] = (np.sum(img[i: i + kernel.shape[0], j: j + kernel.shape[0]] * kernel))
    img_filter = gray_imp(img)
    return signal.convolve2d(img_filter, kernel)


def gaussian_blur(img0: np.ndarray, kernel: np.ndarray = np.array([[0.0625, 0.125, 0.0625],
                                                                   [0.125, 0.25, 0.125],
                                                                   [0.0625, 0.125, 0.0625]])):
    # s = 1
    # cols = (img.shape[0] - kernel.shape[0]) // s + 1
    # rows = (img.shape[1] - kernel.shape[0]) // s + 1
    #
    # lst = np.empty((cols, rows), dtype="int32")

    # convolution implement but it slower. Now I use
    # for i in range(0, cols * s, s):
    #     for j in range(0, rows * s, s):
    #         lst[i // s, j // s] = (np.sum(img[i: i + kernel.shape[0], j: j + kernel.shape[0]] * kernel))
    
    
    """ Remove noise """
    r, g, b = cv2.split(img0)
    r = signal.convolve2d(r, kernel)
    r = np.pad(r, 2)
    g = signal.convolve2d(g, kernel)
    g = np.pad(g, 2)
    b = signal.convolve2d(b, kernel)
    b = np.pad(b, 2)
    return cv2.merge([r, g, b])

    # return cv2.GaussianBlur(img0, (3, 3), 0)


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
