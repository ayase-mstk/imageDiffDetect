"""
画像比較して異なる箇所を赤枠で囲む
"""

import cv2
import os
from opencv_japanese import imread, imwrite
import numpy as np

dirname =  os.path.dirname(__file__)

#画像読み込み
img_1 = imread(dirname + '/../image1/diff_3_1.png')
img_2 = imread(dirname + '/../image1/diff_3_2.png')

height = img_2.shape[0]
width = img_2.shape[1]

img_1 = cv2.resize(img_1 , (int(width), int(height)))

img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

#画像を引き算
img_diff = cv2.absdiff(img_1_gray, img_2_gray)

#2値化
#ret2,img_th = cv2.threshold(img_diff,40,255,cv2.THRESH_BINARY)
ret2,img_th = cv2.threshold(img_diff,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) # histogramを使って2値化

#輪郭を検出
contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#閾値以上の差分を四角で囲う
for i,cnt in enumerate(contours):
    x, y, width, height = cv2.boundingRect(cnt)
    if width > 20 or height > 20:
        cv2.rectangle(img_1, (x, y), (x+width, y+height), (0, 0, 255), 1)

#画像を生成
imwrite(dirname + "/output/diff_neko_image_otsu.jpg", img_1)

