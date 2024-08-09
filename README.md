# 画像比較プログラム
同じ場所で別の時刻の2つの画像を比較して、差分を示す画像を生成・表示するプログラムです。
<!-- シールド一覧 -->
<!-- 該当するプロジェクトの中から任意のものを選ぶ-->
<p style="display: inline">
  <!-- ソースコードで使われている技術 -->
  <img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">
  <img src="https://img.shields.io/badge/-OpenCV-5C3EE8.svg?logo=opencv&style=flat">
  <img src="https://img.shields.io/badge/-NumPy-013243.svg?logo=numpy&style=flat">

</p>

## 差分画像の生成
２種類の差分画像が生成されます
1. 差分が赤色で埋められた画像
2. 差分が緑の枠でマークされた画像

## 差分画像の表示
プログラムを実行すると差分が緑の枠でマークされた画像が表示されます<br>
枠にマウスを置くと元の差分が表示されます<br>
枠をクリックすると拡大画像が表示され、それをクリックするとこちらも元の差分が表示されます<br>

# 準備
python3をインストール
```
pip install --upgrade pip
pip install pillow
pip install opencv-python
pip install numpy
pip install scikit-image
```

# 使い方
```
git clone git@github.com:ayase-mstk/imageDiffDetect.git
cd release
python3 main.py
```

'image/' ディレクトリに画像ファイルを追加し、main関数で指定すれば好きな画像を比較できます
