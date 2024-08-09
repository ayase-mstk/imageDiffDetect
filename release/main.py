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
    img1, img2 = pre.bilateralNoiseFilter(img1, img2)
    #gaus1, gaus2 = pre.gaussianNoiseFilter(img1, img2)
    #median1, median2 = pre.medianNoiseFilter(img1, img2)
    win_utils.displaySideBySide(img1, img2, 'remove noise')
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "noise"), "bilateral_" + path1), img1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "noise"), "bilateral_" + path2), img2)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "noise"), "gaussian_" + path1), gaus1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "noise"), "gaussian_" + path2), gaus2)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "noise"), "median_" + path1), median1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "noise"), "median_" + path2), median2)

    # alignment
    img1 = pre.alignImage(img1, img2)

    # gray scale
    #avg1, avg2 = pre.averageGrayScale(img1, img2) # 平均法
    #gray1, gray2 = pre.grayScale(img1, img2) # 加重法
    cie1, cie2 = pre.CIEXYZ(img1, img2) # 光度法
    win_utils.displaySideBySide(cie1, cie2, 'gray scale')
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "gray_scale"), "avg_" + path1), avg1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "gray_scale"), "avg_" + path2), avg2)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "gray_scale"), "gray_" + path1), gray1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "gray_scale"), "gray_" + path2), gray2)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "gray_scale"), "cie_xyz_" + path1), cie1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "gray_scale"), "cie_xyz_" + path2), cie2)

    # コントラスト調整
    #even1, even2 = pre.adjustContrastImages(cie1, cie2)
    clahe1, clahe2 = pre.claheImages(cie1, cie2)
    #eq1, eq2 = pre.eqHistImages(cie1, cie2)
    #win_utils.displaySideBySide(even1, even2, 'avarage contrast')
    #win_utils.displaySideBySide(clahe1, clahe2, 'clahe contrast')
    #win_utils.displaySideBySide(eq1, eq2, 'eqHist contrast')
    #intermediate_avg1 = cv2.addWeighted(even1, 0.5, clahe1, 0.5, 0)
    #intermediate_avg2 = cv2.addWeighted(even2, 0.5, clahe2, 0.5, 0)
    #gray1 = cv2.addWeighted(intermediate_avg1, 0.5, eq1, 0.5, 0)
    #gray2 = cv2.addWeighted(intermediate_avg2, 0.5, eq2, 0.5, 0)
    #win_utils.displaySideBySide(intermediate_avg1, intermediate_avg1, 'intermidiate avarage contrast')
    #win_utils.displaySideBySide(final_avg1, final_avg2, 'final avarage contrast')
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "contrast"), "even_" + path1), even1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "contrast"), "even_" + path2), even2)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "contrast"), "clahe_" + path1), clahe1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "contrast"), "clahe_" + path2), clahe2)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "contrast"), "eqHist_" + path1), eq1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "contrast"), "eqHist_" + path2), eq2)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "contrast"), "average_" + path1), final_avg1)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "contrast"), "average_" + path2), final_avg2)

    # 差分検出
    diff_img = dd.diffDetect(clahe1, clahe2)
    #sand_diff_img = dd.sandwichDiffDetect(gray1, gray2)
    #adaptive_diff_img = dd.adaptiveDiffDetect(gray1, gray2)
    #ssim_diff_img = dd.ssimDiffDetect(gray1, gray2)
    #back_diff_img = dd.backgroundDiffDetection(gray1, gray2)
    #hist_cmp_img = dd.histogramComparison(gray1, gray2)
    win_utils.displayImage(diff_img, 'diff')
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "binary_diff"), "otsu_thr_" + path1), diff_img)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "binary_diff"), "sand_thr_" + path1), sand_diff_img)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "binary_diff"), "adaptive_thr_" + path1), adaptive_diff_img)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "binary_diff"), "ssim_thr_" + path1), ssim_diff_img)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "binary_diff"), "background_thr_" + path1), back_diff_img)
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "binary_diff"), "histgram_comparison_" + path1), hist_cmp_img)

    # モルフォロジー演算
    closed_img = post.morphologyRemoveNoise(diff_img)
    win_utils.displayImage(closed_img, 'morphology')
    #cv2.imwrite(os.path.join(os.path.join(OUTPUT_PATH, "noise"), "morphology_" + path1), closed_img)

    # 色付け
    highlighted_img = img_utils.highlightDiff(img1, closed_img)
    circle_img, diff_list = img_utils.markShape(img1, closed_img)

    # ファイル出力
    cv2.imwrite(os.path.join(OUTPUT_PATH, 'highlight_' + path1), highlighted_img)
    cv2.imwrite(os.path.join(OUTPUT_PATH, 'circle_' + path1), circle_img)

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
