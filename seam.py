import cv2 as cv
import numpy as np


def load_img(imgName):
    img = cv.imread(imgName,cv.IMREAD_COLOR)
    return img
# end function


def entropy_sobel(image):
    img = cv.Sobel(image,-1,1,1)
    return img
# end function


def find_vert_seam(entropyImage):
    minImg = np.zeros((entropyImage.shape[0], entropyImage.shape[1]))
    minPath = np.zeros((entropyImage.shape[0], entropyImage.shape[1]))
    for x in range(0, entropyImage.shape[1]-1):
        minImg[0,x] = entropyImage[0, x, 0] + entropyImage[0, x, 1] + entropyImage[0, x, 2]
        x = x + 1

    # going top to bottom
    for y in range(1, entropyImage.shape[0]-1):
        for x in range(0,entropyImage.shape[1]-1):
            pixent = int(entropyImage[y, x, 0]) + int(entropyImage[y, x, 1]) + int(entropyImage[y, x, 2])
            minent, minpathVal = find_min(entropyImage.shape[0],x,y,minImg)
            minPath[y, x] = minpathVal
            minImg[y, x] = minent + pixent
            #print(minImg[y, x])

    return minImg, minPath
# end function


def find_min(maxSize, curPlaceX, curPlaceY, minImg):
    values = [999999, 999999, 999999]
    for ctr in range(-1,1):
        if 0 <= ctr + curPlaceX < maxSize:
            values[ctr+1] = int(minImg[curPlaceY-1, ctr+curPlaceX])
    # if values[0] < values[1]:
    #     index = -1
    #     if values[0] > values[2]:
    #         index = 1
    # else:
    #     index = 0
    #     if values[1] > values[2]:
    #         index = 1
    return min(values), values.index(min(values)) - 1
    #return values[index+1], index
# end function


image = load_img("japan.jpg")
entropy = entropy_sobel(image)
minVals, minPath = find_vert_seam(entropy)
