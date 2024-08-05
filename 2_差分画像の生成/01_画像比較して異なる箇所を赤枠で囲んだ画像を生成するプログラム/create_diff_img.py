import cv2
import os
import numpy as np
from opencv_japanese import imread, imwrite

def process_images(img1_path, img2_path, output_path):
    # 画像の読み込み
    img_1 = imread(img1_path)
    img_2 = imread(img2_path)
        
    height = img_2.shape[0]
    width = img_2.shape[1]
        
    # 画像のリサイズ
    img_1 = cv2.resize(img_1, (int(width), int(height)))

    # グレースケール変換
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # 画像の差分を計算
    img_diff = cv2.absdiff(img_1_gray, img_2_gray)

    # 2値化
    ret2, img_th = cv2.threshold(img_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                                                                        
    # 輪郭検出
    contours, _ = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                                                                                    
    # 差分を四角で囲む
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 or h > 20:
            cv2.rectangle(img_1, (x, y), (x+w, y+h), (0, 0, 255), 1)

    # 画像を保存
    imwrite(output_path, img_1)

def batch_process_image_pairs(image_pairs, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, (img1_path, img2_path) in enumerate(image_pairs):
        output_path = os.path.join(output_dir, f"diff_image_{idx+1}.jpg")
        process_images(img1_path, img2_path, output_path)

if __name__ == "__main__":
    image_pairs = []
    while True:
        img1_path = input("1つ目の画像ファイルのパスを入力してください (終了するには 'exit' と入力): ")
        if img1_path.lower() == 'exit':
            break
     
        img2_path = input("2つ目の画像ファイルのパスを入力してください: ")
        
        # 画像ペアをリストに追加
        image_pairs.append((img1_path, img2_path))
        
    # 出力ディレクトリ
    output_dir = './output'

    batch_process_image_pairs(image_pairs, output_dir)

