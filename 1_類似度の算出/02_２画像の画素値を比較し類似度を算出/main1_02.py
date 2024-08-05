"""
２画像の画素値を比較し類似度を算出
"""

import cv2, os
import numpy as np
from opencv_japanese import imread

dirname =  os.path.dirname(__file__)

image1 = imread(dirname + '/1_1.png')
image2 = imread(dirname + '/1_2.png')
image3 = imread(dirname + '/2.png')
image4 = imread(dirname + '/3.png')

height = image1.shape[0]
width = image1.shape[1]

img_size = (int(width), int(height))

# 比較するために、同じサイズにリサイズしておく
image1 = cv2.resize(image1, img_size)
image2 = cv2.resize(image2, img_size)
image3 = cv2.resize(image3, img_size)
image4 = cv2.resize(image4, img_size)

#画素数が一致している割合を計算
print("「1_1.png」と「1_2.png」の類似度：" + str(np.count_nonzero(image1 == image2) / image2.size))
print("「1_1.png」と「2.png」の類似度：" + str(np.count_nonzero(image1 == image3) / image3.size))
print("「1_1.png」と「3.png」の類似度：" + str(np.count_nonzero(image1 == image4) / image4.size))
