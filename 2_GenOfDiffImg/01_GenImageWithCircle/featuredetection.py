import cv2
import os


CWD = os.path.dirname(__file__)

# 画像読み込み
img1 = cv2.imread(CWD + '/../image/image3.png')
img2 = cv2.imread(CWD + '/../image/image4.png')

# ORB
#detector = cv2.ORB_create()
#detector = cv2.AgastFeatureDetector_create() # 特徴量は出さない
detector = cv2.AKAZE_create()

# 特徴点と特徴量を検出
keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

# マッチング器の作成とマッチング
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#matches = bf.match(descriptors1, descriptors2)
#print(len(matches))

# k近傍によるマッチング
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
print(len(matches))
#matches = sorted(matches, key=lambd x: s.distance)

# ratioテストを行う
good_matches = []
for first, second in matches:
    if first.distance < 0.75 * second.distance:
        good_matches.append(first)

# 画像への特徴点の書き込み
out = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 画像出力
cv2.imshow('Matches', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
