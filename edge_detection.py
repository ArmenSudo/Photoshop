import numpy as np
from scipy import signal
import gray_filter


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
    img_filter = gray_filter.gray_imp(img)
    return signal.convolve2d(img_filter, kernel)
