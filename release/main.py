import cv2
import os
import sys

import numpy as np
import pre_process as pre 
import post_process as post
import image_utils as img_utils
import window_utils as win_utils
import diff_detect_library as dd

"""
global 変数
"""
CWD = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(CWD, "output")
IMG_DIR = os.path.abspath(os.path.join(CWD, "../image/"))
WINDOW_NAME = "DIFF_IMAGE"


def outputDiff(path1, path2):
    # 画像読み込み
    ori1 = cv2.imread(os.path.join(IMG_DIR, path1))
    ori2 = cv2.imread(os.path.join(IMG_DIR, path2))

    # 画像のリサイズ
    img1, img2 = img_utils.resizeImages(ori1, ori2)

    # noise 除去
    img1, img2 = pre.removeNoiseBefore(img1, img2)
    win_utils.displaySideBySide(img1, img2, 'remove noise')

    # alignment
    img1 = pre.alignImage(img1, img2)

    # gray scale
    #gray1, gray2 = pre.averageGrayScale(img1, img2) # 平均法
    #gray1, gray2 = pre.grayScale(img1, img2) # 加重法
    gray1, gray2 = pre.CIEXYZ(img1, img2) # 光度法
    win_utils.displaySideBySide(gray1, gray2, 'gray scale')

    # コントラスト調整
    even1, even2 = pre.adjustContrastImages(gray1, gray2)
    clahe1, clahe2 = pre.claheImages(gray1, gray2)
    eq1, eq2 = pre.eqHistImages(gray1, gray2)
    win_utils.displaySideBySide(even1, even2, 'avarage contrast')
    win_utils.displaySideBySide(clahe1, clahe2, 'clahe contrast')
    win_utils.displaySideBySide(eq1, eq2, 'eqHist contrast')
    intermediate_avg1 = cv2.addWeighted(even1, 0.5, clahe1, 0.5, 0)
    final_avg1 = cv2.addWeighted(intermediate_avg1, 0.5, eq1, 0.5, 0)
    intermediate_avg2 = cv2.addWeighted(even2, 0.5, clahe2, 0.5, 0)
    final_avg2 = cv2.addWeighted(intermediate_avg2, 0.5, eq2, 0.5, 0)
    win_utils.displaySideBySide(intermediate_avg1, intermediate_avg1, 'intermidiate avarage contrast')
    win_utils.displaySideBySide(final_avg1, final_avg2, 'final avarage contrast')

    # 差分検出
    diff_img = dd.diffDetect(gray1, gray2)
    #diff_img = dd.sandwichDiffDetect(gray1, gray2)
    #diff_img = dd.adaptiveDiffDetect(gray1, gray2)
    #diff_img = dd.ssimDiffDetect(gray1, gray2)
    #diff_img = dd.backgroundDiffDetection(gray1, gray2)
    win_utils.displayImage(diff_img, 'diff')

    # モルフォロジー演算
    closed_img = post.morphologyRemoveNoise(diff_img)
    win_utils.displayImage(closed_img, 'morphology')

    # 色付け
    highlighted_img = img_utils.highlightDiff(img1, closed_img)
    #circle_img, diff_list = img_utils.markShape(img1, closed_img, 'circle')
    circle_img, diff_list = img_utils.markShape(img1, closed_img)

    # ファイル出力
    cv2.imwrite(os.path.join(OUTPUT_PATH, 'highlight_' + path1), highlighted_img)
    cv2.imwrite(os.path.join(OUTPUT_PATH, 'circle_' + path1), circle_img)

    # イベント登録
    #cv2.setMouseCallback(WINDOW_NAME, win_utils.on_mouse, param=(circle_img.copy(), circle_img, img2, diff_list))

    # 画像表示
    win_utils.displayImage(circle_img, WINDOW_NAME, param=(circle_img.copy(), circle_img, img2, diff_list))



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
