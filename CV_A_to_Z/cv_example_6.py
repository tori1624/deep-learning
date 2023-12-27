# https://wikidocs.net/book/10740

# 1) 이미지 불러오기 및 픽셀 값 조작
import cv2

# 이미지 불러오기
image = cv2.imread("image.jpg")

# 이미지 크기 출력
height, width, channel = image.shape
print("Image shape:", height, width, channel)

# 이미지 픽셀 값 출력
for i in range(height):
    for j in range(width):
        pixel = image[i, j]
        print("Pixel value at position ({},{}) : {}".format(i, j, pixel))


# 2) 이미지 필터링
import cv2

# 이미지 불러오기
image = cv2.imread("image.jpg")

# 이미지 필터링
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# 결과 이미지 출력
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 3) 이미지 간의 차이 계산
import cv2

# 이미지 불러오기
image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

# 이미지 크기 맞추기
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# 이미지 간 차이 계산
diff = cv2.absdiff(image1, image2)

# 차이 시각화
cv2.imshow("Difference Image", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
