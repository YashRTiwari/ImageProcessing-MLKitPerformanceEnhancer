from cv2 import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


original_image = cv2.imread("images/darlk2.jpg")
bw_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
height, width = bw_image.shape
print(width, height)

def saveImage(image, name):
    cv2.imwrite("ouput/" + name + ".jpg", image)

def kernel(kernelSize):
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    return kernel

def showHistogram(i):
    plt.hist(i.ravel(), 256, [0, 256])
    plt.show()

def showResizedImage(original_image, title="Image", scale_percent=35):
    width = int(original_image.shape[1] * scale_percent / 100)
    height = int(original_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def applyDilation(mask, kernelSize=5):
    k = kernel(kernelSize)
    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            # Comparing origin
            if mask[x][y] == k[0][0]*255:
                mask[x ][y ] = 255
            else:
                mask[x][y] = 0
    return mask


def applyErosion(mask,  kernelSize=2):
    k = kernel(kernelSize)
    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            condition = True
            for i in range(0, k.shape[0]):
                for j in range(0, k.shape[1]):
                    if x+i < mask.shape[0] and y+j < mask.shape[1]:
                        if mask[x + i][y + j] != k[i][j]*255:
                            condition = False
            if condition == True:
                mask[x][y] = 255
            else:
                mask[x][y] = 0

    return mask


def adaptiveMeanT(image):
    new_image = np.zeros(shape=image.shape)
    ks = 11
    C = 4
    for i in range(ks, image.shape[0] - ks):
        for j in range(ks, image.shape[1] - ks):
            mx = image[i-ks:i+ks+1, j-ks:j+ks+1]
            if image[i][j] < np.mean(mx) - C:
                new_image[i][j] = 0
            else:
                new_image[i][j] = 255
    return new_image

def niblack(image):
    new_image = np.zeros(shape=image.shape)
    ks = 7
    C = -0.2
    for i in range(ks, image.shape[0] - ks):
        for j in range(ks, image.shape[1] - ks):
            mx = image[i - ks:i + ks + 1, j - ks:j + ks + 1]
            if image[i][j] < np.mean(mx)+(C)*np.std(mx):
                new_image[i][j] = 0
            else:
                new_image[i][j] = 255
    return new_image


def sauvola(image):
    new_image = np.zeros(shape=image.shape)
    ks = 9
    C = 0.3
    for i in range(ks, image.shape[0] - ks):
        for j in range(ks, image.shape[1] - ks):
            mx = image[i - ks:i + ks + 1, j - ks:j + ks + 1]
            formula = np.mean(mx) * (1 + C *((np.std(mx)/128 ) - 1))
            if image[i][j] < formula:
                new_image[i][j] = 0
            else:
                new_image[i][j] = 255
    return new_image


