import cv2
import numpy as np

p1im1 = cv2.imread('./images/p1im1.png', cv2.IMREAD_COLOR)
p1im2 = cv2.imread('./images/p1im2.png', cv2.IMREAD_COLOR)
p1im3 = cv2.imread('./images/p1im3.png', cv2.IMREAD_COLOR)
p1im4 = cv2.imread('./images/p1im4.png', cv2.IMREAD_COLOR)
p1im5 = cv2.imread('./images/p1im5.png', cv2.IMREAD_COLOR)
p1im6 = cv2.imread('./images/p1im6.png', cv2.IMREAD_GRAYSCALE)

def gammaCorrection(image, gamma) :
    newImage = np.power(image / np.max(image), 1 / gamma)
    return newImage

def histogramEqualizationYCrCb(sourceImage) :
    ycrcb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(ycrcb)
    # Histogram processing: Applied only to the intensity component
    image = y
    height = image.shape[0]
    width = image.shape[1]
    histogram = np.zeros(256)

    for i in range(height):
        for j in range(width):
            histogram[image[i , j]] += 1

    cummulativeSum = np.zeros(256)
    for i in range(len(histogram)):
        if i == 0:
            cummulativeSum[i] = histogram[i]
        else:
            cummulativeSum[i] = cummulativeSum[i - 1] + histogram[i]

    normalized = np.zeros(256)
    for i in range(len(normalized)):
        normalized[i] = cummulativeSum[i] * 255 / cummulativeSum[255]

    newImage = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            newImage[i, j] = normalized[image[i,j]]
    y = newImage
    merged = cv2.merge((y, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCR_CB2BGR)

def histogramEqualizationHSV(sourceImage) :
    hsv = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Histogram processing: Applied only to the intensity component
    image = v
    height = image.shape[0]
    width = image.shape[1]
    histogram = np.zeros(256)

    for i in range(height):
        for j in range(width):
            histogram[image[i , j]] += 1

    cummulativeSum = np.zeros(256)
    for i in range(len(histogram)):
        if i == 0:
            cummulativeSum[i] = histogram[i]
        else:
            cummulativeSum[i] = cummulativeSum[i - 1] + histogram[i]

    normalized = np.zeros(256)
    for i in range(len(normalized)):
        normalized[i] = cummulativeSum[i] * 255 / cummulativeSum[255]

    newImage = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            newImage[i, j] = normalized[image[i,j]]
    v = newImage
    merged = cv2.merge((h, s, v))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

def convolution2d(image, filter):
    height = image.shape[0]
    width = image.shape[1]
    b, g, r = cv2.split(image)
    # Linear spatial filtering: Applied to individual RGB components
    row, col = filter.shape
    if (row == col):
        border = int(row / 2)
        b = convolution(b, height, width, border, filter, row)
        g = convolution(g, height, width, border, filter, row)
        r = convolution(r, height, width, border, filter, row)
        merged = cv2.merge((b, g, r))
    return merged

def convolution(image, height, width, border, filter, row):
    newImage = np.zeros(image.shape)
    for i in range(border, height - border):
        for j in range(border, width - border):
            newImage[i][j] = np.sum(image[i-border : i+row-border, j-border : j+row-border] * filter)
    max = np.max(newImage)
    min = np.min(newImage)
    if min < 0: 
        result = np.uint8((newImage + -min)* 255 / np.max(newImage + -min))
    else:
        result = np.uint8(newImage * 255 / np.max(newImage))
    return result

def gaussian(x, y, s, t, gxy, gst, sigmaD, sigmaG):
    d = ((x - s) ** 2 + (y - t) ** 2) / (2 * (sigmaD ** 2))
    g = ((gxy - gst) ** 2) / (2 * (sigmaG ** 2))
    gaussian = np.exp(-d - g)
    return gaussian

def bilateralFilter(image, size, sigmaD, sigmaG):
    height = image.shape[0]
    width = image.shape[1]
    newImage = np.zeros(image.shape)
    border = int(size / 2)
    for x in range(border, height - border):
        for y in range(border, width - border):
            weight = 0
            for i in range(size):
                for j in range(size):
                    s = x + i - border
                    t = y + j - border
                    gxy = image[x][y]
                    gst = image[s][t]
                    newImage[x][y] = newImage[x][y] + gaussian(x, y, s, t, gxy, gst, sigmaD, sigmaG) * gst
                    weight = weight + gaussian(x, y, s, t, gxy, gst, sigmaD, sigmaG)
            newImage[x][y] = newImage[x][y] / weight
    return np.uint8(newImage)

def colorCorrectionHSV(sourceImage, hn, sn, vn):
    hsv = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    height = sourceImage.shape[0]
    width = sourceImage.shape[1]

    for i in range(height):
        for j in range(width):
            if h[i, j] + hn > 179:
                h[i, j] = 179
            elif h[i, j] + hn < 0:
                h[i, j] = 0
            else:
                h[i, j] += hn
            if s[i, j] + sn > 255:
                s[i, j] = 255
            elif s[i, j] + sn < 0:
                s[i, j] = 0
            else:
                s[i, j] += sn
            if v[i, j] + vn > 255:
                v[i, j] = 179
            elif v[i, j] + vn < 0:
                v[i, j] = 0
            else:
                v[i, j] += vn
            
    merged = cv2.merge((h, s, v))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

secondDerivative = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
])

# p1im1_result = gammaCorrection(p1im1, 0.4)
# cv2.imshow('1', p1im1_result)
# cv2.waitKey(0)

# p1im2_result = gammaCorrection(p1im2, 1.7)
# cv2.imshow('2', p1im2_result)
# cv2.waitKey(0)

# p1im4_result = convolution2d(p1im4, secondDerivative)
# cv2.imshow('4', p1im4_result)
# cv2.waitKey(0)

# p1im5_result = histogramEqualizationYCrCb(p1im5)
# cv2.imshow('5-1', p1im5_result)
# cv2.waitKey(0)

# p1im5_result = histogramEqualizationHSV(p1im5)
# cv2.imshow('5-2', p1im5_result)
# cv2.waitKey(0)

# p1im6_result = bilateralFilter(p1im6, 6, 50, 50)
# cv2.imshow('6', p1im6_result)
# cv2.waitKey(0)