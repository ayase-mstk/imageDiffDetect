import cv2
import numpy as np
import argparse
import main


ZOOM_WINDOW_NAME = 'Zoomed'

def on_mouse(event, x, y, flags, param):
    original_image, image1, image2, diff_list = param

    # 画像を押したら赤枠の拡大画像を表示
    if event == cv2.EVENT_LBUTTONDOWN:
        for diff in diff_list:
            dx, dy, dw, dh = diff
            if dx < x < dx + dw and dy < y < dy + dh:
                zoomed = original_image[dy:dy+dh, dx:dx+dw].copy()
                zoomed2 = image2[dy:dy+dh, dx:dx+dw]
                zoomed = cv2.resize(zoomed, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                zoomed2 = cv2.resize(zoomed2, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                cv2.imshow(ZOOM_WINDOW_NAME, zoomed)
                cv2.setMouseCallback(ZOOM_WINDOW_NAME, on_zoomed_mouse, [zoomed, zoomed2])
                return
        cv2.destroyWindow(ZOOM_WINDOW_NAME)

    # 画像にマウスを置いたら赤枠が差分にさし変わる
    if event == cv2.EVENT_MOUSEMOVE:
        image_changed = False
        for diff in diff_list:
            dx, dy, dw, dh = diff
            if dx < x < dx + dw and dy < y < dy + dh:
                image1[dy:dy+dh, dx:dx+dw] = image2[dy:dy+dh, dx:dx+dw]
                image_changed = True
                break

        if not image_changed:
            image1[:] = original_image[:]

        cv2.imshow(main.WINDOW_NAME, image1)
        image_changed = False


def on_zoomed_mouse(event, x, y, flags, param):
    zoomed_img1, zoomed_img2 = param

    # mouseを置いたら画像を差し替えて、クリックしたら戻す
    if event == cv2.EVENT_MOUSEMOVE:
        cv2.imshow(ZOOM_WINDOW_NAME, zoomed_img2)
    elif event == cv2.EVENT_LBUTTONDOWN:
        cv2.imshow(ZOOM_WINDOW_NAME, zoomed_img1)



def displaySideBySide(img1, img2, name='SideBySide'):
    if not isDebugOn():
        return

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
    cv2.imshow(name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def displayImage(img, name=main.WINDOW_NAME, param=None):
    if not isDebugOn() and name != main.WINDOW_NAME:
        return


    while True:
        cv2.imshow(name, img)
        if name == main.WINDOW_NAME:
            cv2.setMouseCallback(main.WINDOW_NAME, on_mouse, param=param)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def isDebugOn():
    # ArgumentParserオブジェクトの作成
    parser = argparse.ArgumentParser(description="Script with a flag.")

    # フラッグオプションを追加
    parser.add_argument('--debug', action='store_true', help='Enable the global flag')

    # 引数をパース
    args = parser.parse_args()

    # フラッグが指定されていればグローバル変数をTrueに設定
    if args.debug:
        return True
    return False


def settingWindow(win_h, win_w):
    # window 作成
    cv2.namedWindow(main.WINDOW_NAME)
    # window size 指定
    cv2.resizeWindow(main.WINDOW_NAME, win_w, win_h)

