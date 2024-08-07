# image_utils.py

import cv2
import sys
import os
import numpy as np

CWD = os.path.dirname(__file__)

def removeNoiseBefore(img1, img2):
    denoised_img1 = cv2.bilateralFilter(img1, d=3, sigmaColor=100, sigmaSpace=100)
    denoised_img2 = cv2.bilateralFilter(img2, d=3, sigmaColor=100, sigmaSpace=100)

    return denoised_img1, denoised_img2

def grayScale(img1, img2):
    return cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

def CIEXYZ(img1, img2):
    xyz_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2XYZ)
    xyz_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2XYZ)

    y_channel1 = xyz_img1[:,:,1]
    y_channel2 = xyz_img2[:,:,1]

    return y_channel1, y_channel2

def grayScaleGammaCorrection(img_bgr1, img_bgr2):
    gamma22LUT  = np.array([pow(x/255.0, 2.2) * 255 for x in range(256)], dtype='uint8')
    gamma045LUT = np.array([pow(x/255.0, 1.0/2.2) * 255 for x in range(256)], dtype='uint8')

    img_bgrL1 = cv2.LUT(img_bgr1, gamma22LUT) # sRGB -> linear
    img_bgrL2 = cv2.LUT(img_bgr2, gamma22LUT) # sRGB -> linear

    img_grayL1 = cv2.cvtColor(img_bgrL1, cv2.COLOR_BGR2GRAY)
    img_grayL2 = cv2.cvtColor(img_bgrL2, cv2.COLOR_BGR2GRAY)

    ##gray1 = np.clip(pow(img_grayL1 / 255.0, 1.0 / 2.2) * 255, 0, 255).astype(np.uint8)
    ##gray2 = np.clip(pow(img_grayL2 / 255.0, 1.0 / 2.2) * 255, 0, 255).astype(np.uint8)
    #gray1 = pow(img_grayL1, 1.0/2.2) * 255 # linear -> sRGB
    #gray2 = pow(img_grayL2, 1.0/2.2) * 255 # linear -> sRGB
    gray1 = cv2.LUT(img_grayL1, gamma045LUT) # linear -> sRGB
    gray2 = cv2.LUT(img_grayL2, gamma045LUT) # linear -> sRGB
    #gray1, _ = cv2.decolor(img_bgr1)
    #gray2, _ = cv2.decolor(img_bgr2)

    return gray1, gray2


def equalizeHistogram(img1, img2):
    equalized1 = cv2.equalizeHist(img1)
    equalized2 = cv2.equalizeHist(img2)

    return equalized1, equalized2


def alignImage(img1, img2):
    # AKAZE特徴点検出器の作成と特徴点の検出:
    detector = cv2.AKAZE_create()
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    match_img = cv2.drawKeypoints(img1, keypoints1, None, color=(0,255,0), flags=0)
    cv2.imshow('keypoints', match_img)
                    
    # 特徴量のマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # k近傍によるマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    print(len(matches))

    # ratioテストを行う
    good_matches = []
    for first, second in matches:
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    # マッチした特徴点の抽出
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                                    
    # ホモグラフィー行列の計算
    # RANSACアルゴリズムを用いて、外れ値を除外
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ホモグラフィー行列に基づき、画像の変換
    if len(img2.shape) == 2:
        height, width = img2.shape
    elif len(img2.shap) == 3:
        height, width, _ = img2.shape
    aligned_img = cv2.warpPerspective(img1, H, (width, height))

    return aligned_img

def resizeImages(img1, img2):
    height = img1.shape[0]
    width = img1.shape[1]
    img_size = (int(width), int(height))

    # 画像をリサイズする
    img2 = cv2.resize(img2, img_size)

    return img1, img2


def edgeEnhance(img1, img2):
    threshold1 = 0
    threshold2 = 360
    return cv2.Canny(img1, threshold1, threshold2), cv2.Canny(img2, threshold1, threshold2)


def morphologyRemoveNoise(binary_mask):
    # ノイズ除去のためのカーネル（サイズは適宜調整）
    kernel = np.ones((3, 3), np.uint8)

    # モルフォロジー演算（オープニング）
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # モルフォロジー演算（クロージング）
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed



def highlightDiff(img1, closed_img):
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    # カラーマスクを使って赤色を付ける
    color_diff = np.zeros_like(img1_color)
    color_diff[closed_img == 255] = [0, 0, 255]  # 赤色で表示

    highlighted_img = cv2.addWeighted(img1_color, 0.7, color_diff, 0.3, 0)

    return highlighted_img

    #im_diff_norm = cv2.normalize(im_diff, None, 0, 255, cv2.NORM_MINMAX)
    #im_diff_norm = np.uint8(im_diff_norm)


    ## 単純に差異をそのまま出力する
    #cv2.imwrite(CWD + '/output/01_diff.png', im_diff_norm)

    ## 差異が無い箇所を中心（灰色：128）とし、そこからの差異を示す
    #cv2.imwrite(CWD + '/output/02_diff_center.png', np.clip(im_diff + 128, 0, 255))

    ## 差異が無い箇所を中心（灰色：128）とし、差異を2で割った商にする（差異を-128～128にしておきたいため）
    #im_diff_center = np.floor_divide(im_diff, 2) + 128
    #cv2.imwrite(CWD + '/output/03_diff_center.png', np.clip(im_diff_center, 0, 255))



def markCircle(img1, closed_img):
    #輪郭を検出
    contours, hierarchy = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

     # グルーピング処理
    min_area = 51
    grouped_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            grouped_contours.append(cnt)

    # 閾値以上の差分を四角で囲う
    # 差分のリストを作成
    min_size = 10 
    diff_list = []
    #for _, cnt in enumerate(contours):
    for cnt in grouped_contours:
        x, y, width, height = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > min_area and (width > min_size or height > min_size):
            cv2.rectangle(img1_color, (x, y), (x+width, y+height), (0, 0, 255), 3)
            diff_list.append((x,y,width,height))

    #画像を生成
    return img1_color, diff_list
