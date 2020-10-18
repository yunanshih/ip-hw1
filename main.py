import cv2
import numpy as np

p1im1 = cv2.imread('./images/p1im1.png', cv2.IMREAD_COLOR)
p1im2 = cv2.imread('./images/p1im2.png', cv2.IMREAD_COLOR)
p1im3 = cv2.imread('./images/p1im3.png', cv2.IMREAD_COLOR)
p1im4 = cv2.imread('./images/p1im4.png', cv2.IMREAD_COLOR)
p1im5 = cv2.imread('./images/p1im5.png', cv2.IMREAD_COLOR)
p1im6 = cv2.imread('./images/p1im6.png', cv2.IMREAD_GRAYSCALE)


def histogramEqualization(sourceImage) :
    ycrcb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(ycrcb)
    image = y
    height = image.shape[0]
    width = image.shape[1]
    histogram = [0] * 256

    for i in range(height):
        for j in range(width):
            histogram[image[i , j]] += 1

    cummulativeSum = [0] * 256
    for i in range(len(histogram)):
        if i == 0:
            cummulativeSum[i] = histogram[i]
        else:
            cummulativeSum[i] = cummulativeSum[i - 1] + histogram[i]

    normalized = [0] * 256
    for i in range(len(normalized)):
        normalized[i] = cummulativeSum[i] * 255 / cummulativeSum[255]

    newImage = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            newImage[i, j] = normalized[image[i,j]]
    y = newImage
    merged = cv2.merge((y, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCR_CB2BGR)

def gammaCorrection(sourceImage, gamma) :
    newImage = np.power(sourceImage / np.max(sourceImage), 1 / gamma)
    return newImage

# def gaussian(s, x):
#     gaussian = 1 / (2 * np.pi * (s ** 2)) * np.exp(- (x ** 2) / (2 * s ** 2))
#     return gaussian

def distance(x1, y1, x2, y2):
    return np.sqrt(np.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))

def convolution2d(image, filter):
    height = image.shape[0]
    width = image.shape[1]
    b, g, r = cv2.split(image)
    row, col = filter.shape
    if (row == col):
        border = int(row / 2)
        b = convolution(b, height, width, border, filter, row)
        g = convolution(g, height, width, border, filter, row)
        r = convolution(r, height, width, border, filter, row)
        newImage = cv2.merge((b, g, r))
    return newImage

def convolution(image, height, width, border, filter, row):
    newImage = np.zeros(image.shape)
    for i in range(border, height - border):
        for j in range(border, width - border):
            newImage[i][j] = np.sum(image[i - border : i - border + row, j - border : j - border + row] * filter)
    max = np.max(newImage)
    min = np.min(newImage)
    if min < 0: 
        result = np.uint8(newImage + -min / np.max(newImage + -min) * 255)
    else:
        result = np.uint8(newImage / np.max(newImage) * 255)
    return result

def gaussian(x, y, s, t, gxy, gst, sigmaD, sigmaG):
    d = ((x - y) ** 2 + (s - t) ** 2) / (2 * (sigmaD ** 2))
    g = ((gxy - gst) ** 2) / (2 * (sigmaG ** 2))
    gaussian = np.exp(-d - g)
    return gaussian

def bilateralFilter(image, size, sigmaD, sigmaG):
    height = image.shape[0]
    width = image.shape[1]
    newImage = np.zeros(image.shape)
    border = int(size / 2)
    weight = 0
        
    for x in range(border, height - border):
        for y in range(border, width - border):
            for i in range(size):
                for j in range(size):
                    s = x + i - 2
                    t = y + j - 2
                    gxy = image[x][y]
                    gst = image[s][t]
                    newImage[x][y] = newImage[x][y] + gaussian(x, y, s, t, gxy, gst, sigmaD, sigmaG) * gst
                    weight = weight + gaussian(x, y, s, t, gxy, gst, sigmaD, sigmaG)
            newImage[x][y] = newImage[x][y] / weight
            weight = 0
    return np.uint8(newImage)

secondDerivative = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
])

p1im1_result = histogramEqualization(p1im1)
cv2.imshow('', p1im1_result)
cv2.waitKey(1000)

p1im2_result = gammaCorrection(p1im2, 1.5)
cv2.imshow('', p1im2_result)
cv2.waitKey(1000)

p1im3_result = gammaCorrection(p1im3, 0.5)
cv2.imshow('', p1im3_result)
cv2.waitKey(1000)

p1im4_result = convolution2d(p1im4, secondDerivative)
cv2.imshow('', p1im4_result)
cv2.waitKey(1000)

p1im5_result = histogramEqualization(p1im5)
cv2.imshow('', p1im5_result)
cv2.waitKey(1000)

# p1im6_result = bilateralFilter(p1im6, 5, 10, 10)
# cv2.imshow('', p1im6_result)
# cv2.waitKey(1000)