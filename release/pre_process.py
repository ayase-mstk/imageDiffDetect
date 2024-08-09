import cv2
import numpy as np

"""
ノイズ処理
"""

def bilateralNoiseFilter(img1, img2):
    denoised_img1 = cv2.bilateralFilter(img1, d=5, sigmaColor=100, sigmaSpace=100)
    denoised_img2 = cv2.bilateralFilter(img2, d=5, sigmaColor=100, sigmaSpace=100)

    return denoised_img1, denoised_img2

def gaussianNoiseFilter(img1, img2, ksize=5):
    # ガウシアンフィルタの適用
    gaus_img1 = cv2.GaussianBlur(img1, (ksize, ksize), 0)
    gaus_img2 = cv2.GaussianBlur(img2, (ksize, ksize), 0)

    return gaus_img1, gaus_img2

def medianNoiseFilter(img1, img2, ksize=5):
    # メディアンフィルタの適用
    median1 = cv2.medianBlur(img1, ksize)
    median2 = cv2.medianBlur(img2, ksize)

    return median1, median2


"""
グレースケール変換
"""

def grayScale(img1, img2):
    return cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

def CIEXYZ(img1, img2):
    xyz_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2XYZ)
    xyz_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2XYZ)

    y_channel1 = xyz_img1[:,:,1]
    y_channel2 = xyz_img2[:,:,1]

    return y_channel1, y_channel2

def deColor(img1, img2):
    de_image1, _ = cv2.decolor(img1)
    de_image2, _ = cv2.decolor(img2)
    return de_image1, de_image2

def averageGrayScale(img1, img2):
        # 画像のサイズを取得
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # グレースケール画像の初期化
    gray_img1 = np.zeros((h1, w1), dtype=np.uint8)
    gray_img2 = np.zeros((h2, w2), dtype=np.uint8)

    # 平均法でグレースケール変換
    gray_img1 = np.mean(img1, axis=2).astype(np.uint8)
    gray_img2 = np.mean(img2, axis=2).astype(np.uint8)

    return gray_img1, gray_img2


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


"""
画像アラインメント
"""

def alignImage(img1, img2):
    # AKAZE特徴点検出器の作成と特徴点の検出:
    detector = cv2.AKAZE_create()
    #detector = cv2.ORB_create()
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    # 特徴点表示
    #match_img = cv2.drawKeypoints(img1, keypoints1, None, color=(0,255,0), flags=0)
    #cv2.imshow('keypoints', match_img)
                    
    # 特徴量のマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # k近傍によるマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

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
    elif len(img2.shape) == 3:
        height, width, _ = img2.shape
    aligned_img = cv2.warpPerspective(img1, H, (width, height))

    return aligned_img

"""
コントラスト調整
"""

def splitToSegment(img):
    height, width = img.shape[:2]

    # 画面の分割
    segments = []
    h_segments = np.array_split(img, 5, axis=0)
    for seg in h_segments:
        segments.extend(np.array_split(seg, 5, axis=1))

    return segments


def adjustContrast(img, alpha):
    # コントラスト調整
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return adjusted


def adjustContrastBySegments(img):
    segments = splitToSegment(img)
    #segments2 = splitToSegment(img2)

    adjusted_segments = []
    for i, segment in enumerate(segments):
        # セグメントの平均輝度を計算
        avg_brightness = np.mean(segment)

        if avg_brightness > 200:
            # 外光と仮定してコントラストをかなり下げる
            adjusted_segments.append(adjustContrast(segment, alpha=0.5))
        elif avg_brightness > 90:
            # 比較的明るい場所と仮定してコントラストを少し下げる
            adjusted_segments.append(adjustContrast(segment, alpha=0.8))
        elif avg_brightness > 50:
            # 室内部分と仮定してコントラストを少し上げる
            adjusted_segments.append(adjustContrast(segment, alpha=1.2))
        else:
            # 室内のくらい部分と仮定してコントラストを少し上げる
            adjusted_segments.append(adjustContrast(segment, alpha=1.5))

    # 調整後のセグメント結合
    adjusted_img = np.vstack([
        np.hstack(adjusted_segments[i*5:(i+1)*5])
        for i in range(5)
    ])

    return adjusted_img


def adjustContrastImages(img1, img2):
    return adjustContrastBySegments(img1), adjustContrastBySegments(img2)

def eqHistImages(img1, img2):
    return eqHist(img1), eqHist(img2)

def claheImages(img1, img2):
    return creClahe(img1), creClahe(img2)

def creClahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)

    # ずれによる黒い部分はマスクする
    mask = (img == 0).astype(np.uint8)
    clahe_img[mask == 1] = 0

    return clahe_img 

def eqHist(img):
    eq_img = cv2.equalizeHist(img)

    # ずれによる黒い部分はマスクする
    mask = (img == 0).astype(np.uint8)
    eq_img[mask == 1] = 0

    return eq_img 
