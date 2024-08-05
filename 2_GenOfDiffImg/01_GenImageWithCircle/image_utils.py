# image_utils.py

import cv2
import numpy as np

def align_images(image1, image2):
    # ORB特徴点検出器の作成と特徴点の検出:
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
                    
    # 特徴量のマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
                                    
    # 対応する特徴点の取得
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
                                                
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # ホモグラフィー行列の計算
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # 画像1の変換
    height, width = image2.shape
    aligned_image = cv2.warpPerspective(image1, h, (width, height))
                                                                                            
    return aligned_image


def remove_noise(threshold):
    # ノイズ除去のためのカーネル（サイズは適宜調整）
    kernel = np.ones((5, 5), np.uint8)

    # モルフォロジー演算（オープニング）
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    # モルフォロジー演算（クロージング）
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed
