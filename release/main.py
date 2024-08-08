import cv2
import os

import numpy as np
import pre_process as pre 
import post_process as post
import image_utils as img_utils
import window_utils as win_utils
import diff_detect_library as dd


CWD = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(CWD, "output")
IMG_DIR = os.path.abspath(os.path.join(CWD, "../image/"))
WINDOW_NAME = "DIFF_IMAGE"


def outputDiff(path1, path2):
    # 画像読み込み
    ori1 = cv2.imread(os.path.join(IMG_DIR, path1))
    ori2 = cv2.imread(os.path.join(IMG_DIR, path2))

    # window 作成
    cv2.namedWindow(WINDOW_NAME)

    # window size 指定
    window_height, window_width = img_utils.calcBalancedSize(ori1)
    cv2.resizeWindow(WINDOW_NAME, window_width, window_height)
    # img resize
    img1 = cv2.resize(ori1, (window_width, window_height), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(ori2, (window_width, window_height), interpolation=cv2.INTER_LINEAR)
    win_utils.displaySideBySide(img1, img2, 'original')

    # noise 除去
    img1, img2 = pre.removeNoiseBefore(img1, img2)
    win_utils.displaySideBySide(img1, img2, 'remove noise')

    # alignment
    img1 = pre.alignImage(img1, img2)

    # resize
    #img1, img2 = img_utils.resizeImages(img1, img2)

    # gray scale
    gray1, gray2 = pre.CIEXYZ(img1, img2)
    win_utils.displaySideBySide(gray1, gray2, 'gray scale')

    # コントラスト調整
    gray1, gray2 = pre.adjustContrastImages(gray1, gray2)
    win_utils.displaySideBySide(gray1, gray2, 'contrast')

    # 差分検出
    diff_img = dd.diffDetect(gray1, gray2)
    #diff_img = dd.sandwichDiffDetect(gray1, gray2)
    #diff_img = dd.adaptiveDiffDetect(gray1, gray2)
    #diff_img = dd.ssimDiffDetect(gray1, gray2)
    #diff_img = dd.backgroundDiffDetection(gray1, gray2)
    win_utils.displayImage(diff_img, 'diff')

    # エッジ強調処理
    #img1, img2 = post.edgeEnhance(img1, img2)

    # モルフォロジー演算
    closed_img = post.morphologyRemoveNoise(diff_img)
    win_utils.displayImage(closed_img, 'morphology')

    # 色付け
    highlighted_img = img_utils.highlightDiff(img1, closed_img)
    #circle_img, diff_list = img_utils.markShape(img1, closed_img, 'circle')
    circle_img, diff_list = img_utils.markShape(img1, closed_img)

    cv2.imwrite(os.path.join(OUTPUT_PATH, 'highlight_' + path1), highlighted_img)
    cv2.imwrite(os.path.join(OUTPUT_PATH, 'circle_' + path1), circle_img)

    # イベント登録
    #cv2.setMouseCallback(WINDOW_NAME, win_utils.on_mouse, param=(circle_img.copy(), circle_img, cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), diff_list))

    # 画像表示
    win_utils.displayImage(circle_img)

def main():
    # test_case
    outputDiff('neko1.png', 'neko2.png')
    outputDiff('image3.png', 'image4.png')
    outputDiff('door1.jpg', 'door2.jpg')
    outputDiff('desk1.jpeg', 'desk2.jpeg')
    outputDiff('chess_1.png', 'chess_2.png')

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    main()
