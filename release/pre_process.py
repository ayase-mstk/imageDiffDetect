import cv2
import numpy as np

"""
ノイズ処理
"""

def removeNoiseBefore(img1, img2):
    denoised_img1 = cv2.bilateralFilter(img1, d=3, sigmaColor=100, sigmaSpace=100)
    denoised_img2 = cv2.bilateralFilter(img2, d=3, sigmaColor=100, sigmaSpace=100)

    return denoised_img1, denoised_img2

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
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

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
    h_segments = np.array_split(img, 4, axis=0)
    for seg in h_segments:
        segments.extend(np.array_split(seg, 4, axis=1))

    return segments

def histogramEqualization(img, alpha):
    # ヒストグラム平坦化
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return adjusted
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #img1 = clahe.apply(img1)
    #img2 = clahe.apply(img2)
    #win_utils.displaySideBySide(img1, img2)

def adjustContrast(img):
    segments = splitToSegment(img)
    #segments2 = splitToSegment(img2)

    adjusted_segments = []
    for i, segment in enumerate(segments):
        if i % 2 == 0:
            # 外光と仮定してコントラストを下げる
            adjusted_segments.append(histogramEqualization(segment, alpha=0.8))
        else:
            # 室内部分と仮定してコントラストを上げる
            adjusted_segments.append(histogramEqualization(segment, alpha=1.2))

    # 調整後のセグメント結合
    adjusted_img = np.vstack([
        np.hstack(adjusted_segments[i*4:(i+1)*4])
        for i in range(4)
    ])

    return adjusted_img

def adjustContrastImages(img1, img2):
    return adjustContrast(img1), adjustContrast(img2)


