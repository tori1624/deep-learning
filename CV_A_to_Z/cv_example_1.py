# https://wikidocs.net/book/10740

import cv2

# 이미지 파일 불러오기
image = cv2.imread('image.jpg')

# 이미지 필터링을 위한 커널 설정
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# 이미지 필터링 적용
filtered_image = cv2.filter2D(image, -1, kernel)

# 결과 이미지 출력
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
