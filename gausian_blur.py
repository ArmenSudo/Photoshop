import numpy as np
from scipy import signal
import cv2


def gaussian_blur(img0: np.ndarray, kernel: np.ndarray = np.array([[0.0625, 0.125, 0.0625],
                                                                   [0.125, 0.25, 0.125],
                                                                   [0.0625, 0.125, 0.0625]])):
    """ Remove noise """
    # s = 1
    # cols = (img.shape[0] - kernel.shape[0]) // s + 1
    # rows = (img.shape[1] - kernel.shape[0]) // s + 1
    #
    # lst = np.empty((cols, rows), dtype="int32")

    # convolution implement but it slower. Now I use
    # for i in range(0, cols * s, s):
    #     for j in range(0, rows * s, s):
    #         lst[i // s, j // s] = (np.sum(img[i: i + kernel.shape[0], j: j + kernel.shape[0]] * kernel))

    r, g, b = cv2.split(img0)
    r = signal.convolve2d(r, kernel)
    r = np.pad(r, 2)
    g = signal.convolve2d(g, kernel)
    g = np.pad(g, 2)
    b = signal.convolve2d(b, kernel)
    b = np.pad(b, 2)
    return cv2.merge([r, g, b])

    # return cv2.GaussianBlur(img0, (3, 3), 0)
