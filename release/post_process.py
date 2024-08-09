import cv2
import numpy as np

def edgeEnhance(img1, img2):
    threshold1 = 0
    threshold2 = 360
    return cv2.Canny(img1, threshold1, threshold2), cv2.Canny(img2, threshold1, threshold2)


def morphologyRemoveNoise(binary_mask):
    # ノイズ除去のためのカーネル（サイズは適宜調整）
    kernel = np.ones((5, 5), np.uint8)

    # モルフォロジー演算（オープニング）
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # モルフォロジー演算（クロージング）
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed

