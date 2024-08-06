import cv2
import os
import image_utils as utils


CWD = os.path.dirname(__file__)

# 画像読み込み
img1 = cv2.imread(CWD + '/../image/image3.png')
img2 = cv2.imread(CWD + '/../image/image4.png')

# noise 除去
img1, img2 = utils.removeNoiseBefore(img1, img2)


# histogram equalization
#img1, img2 = utils.equalizeHistogram(img1, img2)

# alignment
img1 = utils.alignImage(img1, img2)

img1, img2 = utils.resizeImages(img1, img2)

# gray scale
img1, img2 = utils.grayScale(img1, img2)

utils.highlightDiff(img1, img2)
utils.markCircle(img1, img2)

# 画像出力
utils.displaySideBySide(img1, img2)
#cv2.imshow('Differences', img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
