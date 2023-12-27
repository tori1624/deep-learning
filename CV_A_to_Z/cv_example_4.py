# https://wikidocs.net/book/10740

import cv2

# 이미지 경로 지정
image_path = 'image.jpg'

# 이미지 불러오기
image = cv2.imread(image_path)

# 이미지 크기 조정
resized_image = cv2.resize(image, (800, 600))

# 이미지 그레이스케일 변환
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# 이미지 필터링
filtered_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 이미지 노출 조절
adjusted_image = cv2.equalizeHist(filtered_image)

# 각각의 전처리된 이미지 보여주기
cv2.imshow('Original', image)
cv2.imshow('Resized', resized_image)
cv2.imshow('Grayscale', gray_image)
cv2.imshow('Filtered', filtered_image)
cv2.imshow('Adjusted', adjusted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
