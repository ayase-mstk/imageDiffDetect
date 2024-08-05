import cv2, os
from opencv_japanese import imread
from skimage.metrics import structural_similarity as ssim

dirname =  os.path.dirname(__file__)

image1 = cv2.imread(dirname + '/1_1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(dirname + '/1_2.png', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread(dirname + '/2.png', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(dirname + '/3.png', cv2.IMREAD_GRAYSCALE)

height = image1.shape[0]
width = image1.shape[1]

img_size = (int(width), int(height))

# 比較するために、同じサイズにリサイズしておく
image1 = cv2.resize(image1, img_size)
image2 = cv2.resize(image2, img_size)
image3 = cv2.resize(image3, img_size)
image4 = cv2.resize(image4, img_size)

# SIMによる画像の類似度を比較する
ssim_1_2 = ssim(image1, image2)
ssim_1_3 = ssim(image1, image3)
ssim_1_4 = ssim(image1, image4)

print(f"「1_1.png」と「1_2.png」のSSIM類似度: {ssim_1_2:.4f}")
print(f"「1_1.png」と「2.png」のSSIM類似度: {ssim_1_3:.4f}")
print(f"「1_1.png」と「3.png」のSSIM類似度: {ssim_1_4:.4f}")
