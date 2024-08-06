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

    displayGrayScale(gray1, gray2)
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
                                    
    ## 対応する特徴点の取得
    #points1 = np.zeros((len(matches), 2), dtype=np.float32)
    #points2 = np.zeros((len(matches), 2), dtype=np.float32)

    # ホモグラフィー行列の計算
    # RANSACアルゴリズムを用いて、外れ値を除外
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ホモグラフィー行列に基づき、画像の変換
    #height, width = img2.shape
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


def removeNoise(threshold):
    # ノイズ除去のためのカーネル（サイズは適宜調整）
    kernel = np.ones((5, 5), np.uint8)

    # モルフォロジー演算（オープニング）
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # モルフォロジー演算（クロージング）
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed


def highlightDiff(img1, img2):
    # ２画像の差異を計算
    im_diff = cv2.absdiff(img1, img2)

    # 差異の閾値を適用してマスクを作成
    threshold = 30
    #gray_diff = cv2.cvtColor(im_diff, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(im_diff, threshold, 255, cv2.THRESH_BINARY)

    # マスクを使って色を付ける
    color_diff = np.zeros_like(img1)
    color_diff[binary_mask == 255] = [0, 0, 255]  # 赤色で表示

    highlighted_img = cv2.addWeighted(img1, 0.7, color_diff, 0.3, 0)
    cv2.imwrite(CWD + '/output/highlight.png', highlighted_img)

    im_diff_norm = cv2.normalize(im_diff, None, 0, 255, cv2.NORM_MINMAX)
    im_diff_norm = np.uint8(im_diff_norm)


    # 単純に差異をそのまま出力する
    cv2.imwrite(CWD + '/output/01_diff.png', im_diff_norm)

    # 差異が無い箇所を中心（灰色：128）とし、そこからの差異を示す
    cv2.imwrite(CWD + '/output/02_diff_center.png', np.clip(im_diff + 128, 0, 255))

    # 差異が無い箇所を中心（灰色：128）とし、差異を2で割った商にする（差異を-128～128にしておきたいため）
    im_diff_center = np.floor_divide(im_diff, 2) + 128
    cv2.imwrite(CWD + '/output/03_diff_center.png', np.clip(im_diff_center, 0, 255))



def markCircle(img1, img2):
    #画像を引き算、その時黒い部分はマスクする
    mask = (img1 == 0).astype(np.uint8)
    img_diff = cv2.absdiff(img1, img2)
    img_diff[mask == 1] = 0

    #2値化
    #ret2,img_th = cv2.threshold(img_diff,40,255,cv2.THRESH_BINARY)
    ret2,img_th = cv2.threshold(img_diff,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) # histogramを使って2値化

    #輪郭を検出
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #閾値以上の差分を四角で囲う
    for i,cnt in enumerate(contours):
        x, y, width, height = cv2.boundingRect(cnt)
        if width > 20 or height > 20:
            cv2.rectangle(img1, (x, y), (x+width, y+height), (0, 0, 255), 1)

    #画像を生成
    cv2.imwrite(CWD + "/output/circle.jpg", img1)



def displayGrayScale(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
            
    # 高さと幅の最大値を計算
    h = max(h1, h2)
    w = max(w1, w2)

    # 画像をパディングしてサイズを合わせる
    if img1.ndim == 3:  # カラー画像の場合
        img1 = np.pad(img1, ((0, h - h1), (0, w - w1), (0, 0)), mode='constant', constant_values=255)
        img2 = np.pad(img2, ((0, h - h2), (0, w - w2), (0, 0)), mode='constant', constant_values=255)
    else:  # グレースケール画像の場合
        img1 = np.pad(img1, ((0, h - h1), (0, w - w1)), mode='constant', constant_values=255)
        img2 = np.pad(img2, ((0, h - h2), (0, w - w2)), mode='constant', constant_values=255)

    # 画像を横に並べる
    combined = np.hstack((img1, img2))

    # 結果を表示
    cv2.imshow('Side by Side', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def displaySideBySide(img1, img2):
    # 画像の高さを揃える
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)

    # 高さが足りない方に余白を追加
    if h1 < h:
        img1 = np.pad(img1, ((0, h - h1), (0, 0), (0, 0)), mode='constant', constant_values=255)
    elif h2 < h:
        img2 = np.pad(img2, ((0, h - h2), (0, 0), (0, 0)), mode='constant', constant_values=255)
 
    # 2つの画像を横に連結
    combined = np.hstack((img1, img2))

    # 結合した画像を表示
    cv2.imshow('Images Side by Side', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
