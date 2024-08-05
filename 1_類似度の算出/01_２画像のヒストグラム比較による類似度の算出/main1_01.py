"""
２画像のヒストグラム比較による類似度の算出
"""

import cv2, os
from opencv_japanese import imread

dirname =  os.path.dirname(__file__)

image1 = cv2.imread(dirname + '/1_1.png')
image2 = cv2.imread(dirname + '/1_2.png')
image3 = cv2.imread(dirname + '/2.png')
image4 = cv2.imread(dirname + '/3.png')
image5 = cv2.imread(dirname + '/../image1/diff_3_1.png')
image6 = cv2.imread(dirname + '/../image1/diff_3_2.png')

height = image1.shape[0]
width = image1.shape[1]

img_size = (int(width), int(height))

# 比較するために、同じサイズにリサイズしておく
image1 = cv2.resize(image1, img_size)
image2 = cv2.resize(image2, img_size)
image3 = cv2.resize(image3, img_size)
image4 = cv2.resize(image4, img_size)
image5 = cv2.resize(image5, img_size)
image6 = cv2.resize(image6, img_size)

# 画像をヒストグラム化する
image1_hist = cv2.calcHist([image1], [2], None, [256], [0, 256])
image2_hist = cv2.calcHist([image2], [2], None, [256], [0, 256])
image3_hist = cv2.calcHist([image3], [2], None, [256], [0, 256])
image4_hist = cv2.calcHist([image4], [2], None, [256], [0, 256])
image5_hist = cv2.calcHist([image5], [2], None, [256], [0, 256])
image6_hist = cv2.calcHist([image6], [2], None, [256], [0, 256])

# ヒストグラムした画像を比較
print("「1_1.png」と「1_2.png」の類似度：" + str(cv2.compareHist(image1_hist, image2_hist, 0)))
print("「1_1.png」と「2.png」の類似度：" + str(cv2.compareHist(image1_hist, image3_hist, 0)))
print("「1_1.png」と「3.png」の類似度：" + str(cv2.compareHist(image1_hist, image4_hist, 0)))
print("「diff_3_0.png」と「diff_3_1.png」の類似度：" + str(cv2.compareHist(image5_hist, image6_hist, 0)))
