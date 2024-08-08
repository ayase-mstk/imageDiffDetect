import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

OUTPUT_PATH = "./output/"

def cv2_kmeans(image, n_clusters=4):
    # 画像をリシェイプして、k-meansクラスタリングの入力データに変換
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # k-meansクラスタリングの条件を設定
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, n_clusters, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)

    # 画像化
    h, w, _ = image.shape
    result_image = np.zeros((h, w, 3), dtype=np.uint8)

    # 各クラスタの色をランダムに設定
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n_clusters)]

    # 各ピクセルにクラスタの色を適用
    labels = labels.flatten()  # 1次元配列に変換
    for i in range(h):
        for j in range(w):
            cluster_idx = labels[i * w + j]
            result_image[i, j] = colors[cluster_idx]

    # 各クラスタの中心点をプロット
    for i in range(n_clusters):
        cx, cy = int(centers[i][0]), int(centers[i][1])
        color = (0, 0, 0)
        cv2.circle(result_image, (cx, cy), 5, color, -1)
        msg = f"({centers[i][0]:.2f},{centers[i][1]:.2f})"
        cv2.putText(result_image, msg, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)

    cv2.imshow("cv2_kmeans", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img1 = cv2.imread('../image/image3.png')
    img2 = cv2.imread('../image/image4.png')

    cv2_kmeans(img1)
    cv2_kmeans(img2)

    highlighted_img, diff, thresh, segmented_img1, segmented_img2 = highlightDiff(img1, img2, threshold=0.5, min_contour_area=500)

    cv2.imshow('Highlighted Differences', highlighted_img)
    cv2.imshow('Diff Image', diff)
    cv2.imshow('Threshold Image', thresh)
    cv2.imshow('Segmented Image 1', segmented_img1)
    cv2.imshow('Segmented Image 2', segmented_img2)

    # Save the segmented images
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    cv2.imwrite(os.path.join(OUTPUT_PATH, 'segmented_img1.png'), segmented_img1)
    cv2.imwrite(os.path.join(OUTPUT_PATH, 'segmented_img2.png'), segmented_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

