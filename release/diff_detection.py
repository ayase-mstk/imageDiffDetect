import cv2
import os


CWD = os.path.dirname(__file__)

# 画像読み込み
img1 = cv2.imread(CWD + '/../image/image3.png')
img2 = cv2.imread(CWD + '/../image/image4.png')

# noise 除去
denoised_img1 = cv2.bilateralFilter(img1, d=3, sigmaColor=100, sigmaSpace=100)
denoised_img2 = cv2.bilateralFilter(img2, d=3, sigmaColor=100, sigmaSpace=100)

# 画像出力
cv2.imshow('Differences', denoised_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
