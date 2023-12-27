# https://wikidocs.net/book/10740

import cv2

# 이미지 불러오기
image = cv2.imread('image.jpg')

# 이미지 크기 변경
resized_image = cv2.resize(image, (300, 300))

# 이미지 회전
rows, cols, _ = resized_image.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated_image = cv2.warpAffine(resized_image, M, (cols, rows))

# 이미지 저장
cv2.imwrite('output.jpg', rotated_image)
