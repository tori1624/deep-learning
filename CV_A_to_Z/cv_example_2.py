# https://wikidocs.net/book/10740

# 1) 이미지 로딩 및 디스플레이
import cv2

# 이미지 로딩
image = cv2.imread('image.jpg')

# 윈도우 창 생성 및 이미지 디스플레이
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', image)

# 키보드 입력 대기
cv2.waitKey(0)

# 윈도우 창 종료
cv2.destroyAllWindows()


# 2) 이미지 필터링
import cv2

# 이미지 로딩
image = cv2.imread('image.jpg')

# 가우시안 블러 적용
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# 윈도우 창 생성 및 이미지 디스플레이
cv2.namedWindow('Blurred Image', cv2.WINDOW_NORMAL)
cv2.imshow('Blurred Image', blurred)

# 키보드 입력 대기
cv2.waitKey(0)

# 윈도우 창 종료
cv2.destroyAllWindows()


# 3) 이미지 이진화
import cv2

# 이미지 로딩
image = cv2.imread('image.jpg')

# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이진화
ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 윈도우 창 생성 및 이미지 디스플레이
cv2.namedWindow('Threshold Image', cv2.WINDOW_NORMAL)
cv2.imshow('Threshold Image', threshold)

# 키보드 입력 대기
cv2.waitKey(0)

# 윈도우 창 종료
cv2.destroyAllWindows()


# 4) 관심 영역 검출
import cv2

# 이미지 로딩
image = cv2.imread('image.jpg')

# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 캐니 에지 검출
edges = cv2.Canny(gray, 50, 150)

# 윈도우 창 생성 및 이미지 디스플레이
cv2.namedWindow('Edges Image', cv2.WINDOW_NORMAL)
cv2.imshow('Edges Image', edges)

# 키보드 입력 대기
cv2.waitKey(0)

# 윈도우 창 종료
cv2.destroyAllWindows()
