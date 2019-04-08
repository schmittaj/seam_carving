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
    for x in range(0, entropyImage.shape[1]):
        minImg[0,x] = entropyImage[0, x, 0] + entropyImage[0, x, 1] + entropyImage[0, x, 2]

    # going top to bottom
    for y in range(1, entropyImage.shape[0]):
        for x in range(0,entropyImage.shape[1]):
            pixent = int(entropyImage[y, x, 0]) + int(entropyImage[y, x, 1]) + int(entropyImage[y, x, 2])
            minent, minpathVal = find_min_vert(entropyImage.shape[1],x,y,minImg)
            minPath[y, x] = minpathVal
            minImg[y, x] = minent + pixent

    return minImg, minPath
# end function


def find_horz_seam(entropyImage):
    minImg = np.zeros((entropyImage.shape[0], entropyImage.shape[1]))
    minPath = np.zeros((entropyImage.shape[0], entropyImage.shape[1]))
    for y in range(0, entropyImage.shape[0]):
        minImg[y,0] = entropyImage[y, 0, 0] + entropyImage[y, 0, 1] + entropyImage[y, 0, 2]

    # going left to right
    for x in range(1, entropyImage.shape[0]):
        for y in range(0,entropyImage.shape[1]):
            pixent = int(entropyImage[y, x, 0]) + int(entropyImage[y, x, 1]) + int(entropyImage[y, x, 2])
            minent, minpathVal = find_min_horz(entropyImage.shape[1],x,y,minImg)
            minPath[y, x] = minpathVal
            minImg[y, x] = minent + pixent

    return minImg, minPath
# end function


def find_min_vert(maxSize, curPlaceX, curPlaceY, minImg):
    values = [999999, 999999, 999999]
    for ctr in range(-1,2):
        if 0 <= ctr + curPlaceX < maxSize:
            values[ctr+1] = int(minImg[curPlaceY-1, ctr+curPlaceX])
    return min(values), values.index(min(values)) - 1
# end function


def find_min_horz(maxSize, curPlaceX, curPlaceY, minImg):
    values = [999999, 999999, 999999]
    for ctr in range(-1,2):
        if 0 <= ctr + curPlaceY < maxSize:
            values[ctr+1] = int(minImg[ctr+curPlaceY, curPlaceX-1])
    return min(values), values.index(min(values)) - 1
# end function


image = load_img("test.jpg")

entropy = entropy_sobel(image)
minVals, minPath = find_vert_seam(entropy)
#cv.imshow("",entropy)
#cv.waitKey(0)


f = open("output.txt", "w")

f.write("Values:" + "\n")
for y in range(0,minPath.shape[0]):
    line = ""
    for x in range(0,minPath.shape[1]):
        showVal = int(entropy[y, x, 0]) + int(entropy[y, x, 1]) + int(entropy[y, x, 2])
        line += "" + str(showVal) + " "
    f.write(line + "\n")

f.write("\n" + "MinVals:" + "\n")
for y in range(0,minPath.shape[0]):
    line = ""
    for x in range(0,minPath.shape[1]):
        line += "" + str(minVals[y, x]) + " "
    f.write(line + "\n")

f.write("\n" + "MinPath:" + "\n")
for y in range(0,minPath.shape[0]):
    line = ""
    for x in range(0,minPath.shape[1]):
        line += "" + str(minPath[y, x]) + " "
    f.write(line + "\n")

f.close()
