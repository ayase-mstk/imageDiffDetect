from skimage.metrics import structural_similarity as ssim
from image_utils import align_images, remove_noise
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

image1 = "../image2/image3.png"
image2 = "../image2/image4.png"


# 画像をグレースケールで読み込む
img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

# 画像のサイズを一致させる
height, width = img1.shape
img2 = resize_image(img2, (width, height))

# 画像のalignment
img1 = align_images(img1, img2)

# SSIMを計算し、SSIMマップを取得
ssim_score, ssim_map = ssim(img1, img2, full=True)

# SIMマップの平滑化で細かい違いをぼかせる
smoothed_ssim_map = cv2.GaussianBlur(ssim_map, (5, 5), 0)

# 2値化
threshold = 0.2
binary_diff = (smoothed_ssim_map < threshold).astype(np.uint8) * 255

# ノイズの除去
closed = remove_noise(binary_diff)

# 元のカラー画像を読み込む
color_img1 = cv2.imread(image1)

# 輪郭を取得
#contours, _ = cv2.findContours(binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours, _ = cv2.findContours(binary_diff, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#contours, _ = cv2.findContours(binary_diff, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
contours, _ = cv2.findContours(binary_diff, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)


# 差異を赤枠で囲む
for contour in contours:
    if cv2.contourArea(contour) > 100:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(color_img1, (x, y), (x+w, y+h), (0, 0, 255), 2)
 
# 結果を表示
cv2.imshow('Differences', color_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
