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
        if 0 < ctr + curPlaceX < maxSize-1: # changed from 0 <= ctr + curPlaceX < maxSize: to ignore edges that will always be 0
            values[ctr+1] = int(minImg[curPlaceY-1, ctr+curPlaceX])
    return min(values), values.index(min(values)) - 1
# end function


def find_min_horz(maxSize, curPlaceX, curPlaceY, minImg):
    values = [999999, 999999, 999999]
    for ctr in range(-1,2):
        if 0 < ctr + curPlaceY < maxSize-1: # changed from 0 <= ctr + curPlaceX < maxSize: to ignore edges that will always be 0
            values[ctr+1] = int(minImg[ctr+curPlaceY, curPlaceX-1])
    return min(values), values.index(min(values)) - 1
# end function


image = load_img("test.jpg")

entropy = entropy_sobel(image)
minValsV, minPathV = find_vert_seam(entropy)
minValsH, minPathH = find_horz_seam(entropy)
#cv.imshow("",entropy)
#cv.waitKey(0)


f = open("output.txt", "w")

f.write("Values:" + "\n")
for y in range(0,minPathV.shape[0]):
    line = ""
    for x in range(0,minPathV.shape[1]):
        showVal = int(entropy[y, x, 0]) + int(entropy[y, x, 1]) + int(entropy[y, x, 2])
        line += "" + str(showVal) + "\t"
    f.write(line + "\n")

f.write("\n" + "MinValsV:" + "\n")
for y in range(0,minPathV.shape[0]):
    line = ""
    for x in range(0,minPathV.shape[1]):
        line += "" + str(minValsV[y, x]) + "\t\t"
    f.write(line + "\n")

f.write("\n" + "MinPathV:" + "\n")
for y in range(0,minPathV.shape[0]):
    line = ""
    for x in range(0,minPathV.shape[1]):
        line += "" + str(minPathV[y, x]) + "\t\t"
    f.write(line + "\n")

f.write("\n" + "MinValsH:" + "\n")
for y in range(0,minPathH.shape[0]):
    line = ""
    for x in range(0,minPathH.shape[1]):
        line += "" + str(minValsH[y, x]) + "\t\t"
    f.write(line + "\n")

f.write("\n" + "MinPathH:" + "\n")
for y in range(0,minPathH.shape[0]):
    line = ""
    for x in range(0,minPathH.shape[1]):
        line += "" + str(minPathH[y, x]) + "\t\t"
    f.write(line + "\n")


f.close()
