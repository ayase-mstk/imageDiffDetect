import cv2
import numpy as np
import main

def on_mouse(event, x, y, flags, param):
    image1, image2, diff_list = param

    if event == cv2.EVENT_MOUSEMOVE:
        for diff in diff_list:
            dx, dy, dw, dh = diff
            if dx < x < dx + dw and dy < y < dy + dh:
                roi = image1[dy:dy+dh, dx:dx+dw]
                zoomed = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Zoomed', zoomed)
                return
        cv2.destroyWindow('Zoomed')


    if event == cv2.EVENT_LBUTTONDOWN:
        for diff in diff_list:
            dx, dy, dw, dh = diff
            if dx < x < dx + dw and dy < y < dy + dh:
                # クリックされた赤枠の部分をもう一つの画像から切り出す
                roi = image2[dy:dy+dh, dx:dx+dw]
                cv2.imshow('Matched Region', roi)
                return
        cv2.destroyWindow('Matched Region')




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



def displayImage(img):
    # 結合した画像を表示
    cv2.imshow(main.WINDOW_NAME, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
