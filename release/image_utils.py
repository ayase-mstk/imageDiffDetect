# image_utils.py

import cv2
import sys
import os
import numpy as np

def resizeImages(img1, img2):
    height = img1.shape[0]
    width = img1.shape[1]
    img_size = (int(width), int(height))

    # 画像をリサイズする
    img2 = cv2.resize(img2, img_size)

    return img1, img2

def highlightDiff(img1, closed_img):
    #img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img1_color = img1

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


def markShape(img1, closed_img, shape='rounded_rect'):
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img1_color = img1

    min_area = 51
    min_size = 10
    diff_list = []

    # グルーピング処理
    grouped_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            grouped_contours.append(cnt)


    for cnt in grouped_contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            if shape == 'circle':
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                area = cv2.contourArea(cnt)
                if radius > min_size // 2:
                    cv2.circle(img1_color, center, radius, (0, 255, 0), 3)
                    diff_list.append((int(x-radius), int(y-radius), int(radius*2), int(radius*2)))
            elif shape == 'rounded_rect':
                x, y, w, h = cv2.boundingRect(cnt)
                if w > min_size or h > min_size:
                    radius = int(min(w, h) * 0.1)  # 角の丸みの半径（調整可能）
                    draw_rounded_rectangle(img1_color, (x, y), (x+w, y+h), (0, 255, 0), 3, radius)
                    diff_list.append((x, y, w, h))

    return img1_color, diff_list



def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius):
    x1, y1 = pt1
    x2, y2 = pt2

    # 角丸の四角形を描画
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    # 四隅に円弧を描画
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)



def calcBalancedSize(img, max_size=800):
    # 画像の元のサイズを取得
    height, width = img.shape[:2]

    # 長い方がmax_sizeになるようにリサイズ比を計算
    if width > height:
        new_width = max_size
        new_height = int((new_width / width) * height)
    else:
        new_height = max_size
        new_width = int((new_height / height) * width)

    return new_height, new_width
