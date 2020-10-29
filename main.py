import cv2
import numpy as np

p1im1 = cv2.imread('./images/p1im1.png', cv2.IMREAD_COLOR)
p1im2 = cv2.imread('./images/p1im2.png', cv2.IMREAD_COLOR)
p1im3 = cv2.imread('./images/p1im3.png', cv2.IMREAD_COLOR)
p1im4 = cv2.imread('./images/p1im4.png', cv2.IMREAD_COLOR)
p1im5 = cv2.imread('./images/p1im5.png', cv2.IMREAD_COLOR)
p1im6 = cv2.imread('./images/p1im6.png', cv2.IMREAD_GRAYSCALE)

def gammaCorrection(image, gamma):
    newImage = np.power(image / np.max(image), gamma)
    return newImage

def histogramEqualizationYCrCb(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(ycrcb)
    # Histogram processing: Applied only to the intensity component
    y = histogramEqualization(y)
    merged = cv2.merge((y, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCR_CB2BGR)

def histogramEqualizationHSV(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Histogram processing: Applied only to the intensity component
    v = histogramEqualization(v)
    merged = cv2.merge((h, s, v))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

def histogramEqualizationBGR(image):
    b, g, r = cv2.split(image)
    b = histogramEqualization(b)
    g = histogramEqualization(g)
    r = histogramEqualization(r)
    merged = cv2.merge((b, g, r))
    return merged

def histogramEqualization(image):
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
    return newImage


def spatialFiltering(image, filter):
    height = image.shape[0]
    width = image.shape[1]
    b, g, r = cv2.split(image)
    # Linear spatial filtering: Applied to individual RGB components
    row, col = filter.shape
    if (row == col):
        border = int(row / 2)
        b = filtering(b, height, width, border, filter, row)
        g = filtering(g, height, width, border, filter, row)
        r = filtering(r, height, width, border, filter, row)
        merged = cv2.merge((b, g, r))
    return merged

def filtering(image, height, width, border, filter, row):
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

def colorCorrectionBGR(image, bn, gn, rn):
    b, g, r = cv2.split(image)
    height = image.shape[0]
    width = image.shape[1]

    for i in range(height):
        for j in range(width):
            if b[i, j] + bn > 255:
                b[i, j] = 255
            elif b[i, j] + bn < 0:
                b[i, j] = 0
            else:
                b[i, j] += bn
            if g[i, j] + gn > 255:
                g[i, j] = 255
            elif g[i, j] + gn < 0:
                g[i, j] = 0
            else:
                g[i, j] += gn
            if r[i, j] + rn > 255:
                r[i, j] = 179
            elif r[i, j] + rn < 0:
                r[i, j] = 0
            else:
                r[i, j] += rn
    return cv2.merge((b, g, r))

sharpening = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
])

p1im1_result = gammaCorrection(p1im1, 2.3)
cv2.imshow('1', p1im1_result)
cv2.waitKey(0)

p1im2_result = gammaCorrection(p1im2, 0.6)
cv2.imshow('2', p1im2_result)
cv2.waitKey(0)

p1im3_result = colorCorrectionBGR(p1im3, 0, 0, -20)
p1im3_result = gammaCorrection(p1im3_result, 2.4)
cv2.imshow('3', p1im3_result)
cv2.waitKey(0)

p1im4_result = spatialFiltering(p1im4, sharpening)
cv2.imshow('4', p1im4_result)
cv2.waitKey(0)

p1im5_result = histogramEqualizationYCrCb(p1im5)
cv2.imshow('5', p1im5_result)
cv2.waitKey(0)

p1im5_result = histogramEqualizationBGR(p1im5)
cv2.imshow('5', p1im5_result)
cv2.waitKey(0)

p1im6_result = bilateralFilter(p1im6, 9, 100, 100)
cv2.imshow('6', p1im6_result)
cv2.waitKey(0)
