import numpy as np


def contrast(img: np.ndarray, value: int = 30):
    img_filter = img.copy()
    img_filter[img_filter < value] = 0
    img_filter[(img_filter >= value) & (img_filter <= 128)] -= value
    img_filter[img_filter > 255 - value] = 255
    img_filter[(img_filter < 255 - value) & (img > 128)] += value
    return img_filter
