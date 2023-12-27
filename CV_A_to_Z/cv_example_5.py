# https://wikidocs.net/book/10740

# 1) 이미지 필터링
import cv2
import numpy as np

def filter_image(image, kernel):
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
image = cv2.imread('input_image.jpg')

filtered_image = filter_image(image, kernel)

cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 2) 객체 탐지
import cv2

def detect_objects(image):
    cascade_path = 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image

image = cv2.imread('input_image.jpg')

detected_image = detect_objects(image)

cv2.imshow('Detected Image', detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
