import cv2
import numpy as np
import image_utils as img_utils
import window_utils as win_utils

from skimage.metrics import structural_similarity as ssim

def diffDetect(img1, img2):
    # コントラスト調整
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)
    #win_utils.displaySideBySide(img1, img2)

    #画像を引き算
    img_diff = cv2.absdiff(img1, img2)

    #2値化
    #under_thresh = 105
    #upper_thresh = 145
    #maxValue = 255
    #th, drop_back = cv2.threshold(img_diff, under_thresh, maxValue, cv2.THRESH_BINARY)
    #th, clarify_born = cv2.threshold(img_diff, upper_thresh, maxValue, cv2.THRESH_BINARY_INV)
    #img_th = np.minimum(drop_back, clarify_born)

    #ret2,img_th = cv2.threshold(img_diff,40,255,cv2.THRESH_BINARY)
    ret2,img_th = cv2.threshold(img_diff,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) # histogramを使って2値化

    #img_th = cv2.adaptiveThreshold(img_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 20)

    # 黒い部分はマスクする
    mask = (img1 == 0).astype(np.uint8)
    img_th[mask == 1] = 0

    return img_th

def ssimDiffDetect(img1, img2):
    # SSIMの計算
    score, diff = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")

    # 差分画像の二値化
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return thresh


def backgroundDiffDetect(img1, img2):
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    fgMask1 = backSub.apply(img1)
    fgMask2 = backSub.apply(img2)

    result = fgMask2

    return result
