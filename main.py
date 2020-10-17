import cv2
import numpy as np

p1im1 = cv2.imread('./images/p1im1.png')
p1im2 = cv2.imread('./images/p1im2.png')
p1im3 = cv2.imread('./images/p1im3.png')
p1im4 = cv2.imread('./images/p1im4.png')
p1im5 = cv2.imread('./images/p1im5.png')
p1im6 = cv2.imread('./images/p1im6.png')


def findHistogram(sourceImage) :
    ycrcb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    image = channels[0]

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
    
    channels[0] = newImage
    
    cv2.merge(channels, ycrcb)
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return result


def getGaussian(s, x) :
    gaussian = 1/ (np.sqrt (2 * np.pi ) * s) * np.e ** ( - ( x**2 / (2 * s ** 2)))
    return gaussian

def correctGamma(sourceImage, gamma) :
    height = sourceImage.shape[0]
    width = sourceImage.shape[1]

    newImage =  np.power(sourceImage / np.max(sourceImage), 1 / gamma)
    return newImage


# p1im1_result = findHistogram(p1im5)
# cv2.imshow('', p1im1_result)
# cv2.waitKey(5000)


p1im2_result = correctGamma(p1im2, 1.5)
cv2.imshow('', p1im2_result)
cv2.waitKey(5000)