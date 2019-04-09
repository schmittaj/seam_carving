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
    minImg = np.zeros((entropyImage.shape[0], entropyImage.shape[1]), dtype=int)
    minPath = np.zeros((entropyImage.shape[0], entropyImage.shape[1]), dtype=int)
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
    minImg = np.zeros((entropyImage.shape[0], entropyImage.shape[1]), dtype=int)
    minPath = np.zeros((entropyImage.shape[0], entropyImage.shape[1]), dtype=int)
    for y in range(0, entropyImage.shape[0]):
        minImg[y,0] = entropyImage[y, 0, 0] + entropyImage[y, 0, 1] + entropyImage[y, 0, 2]

    # going left to right
    for x in range(1, entropyImage.shape[1]):
        for y in range(0,entropyImage.shape[0]):
            pixent = int(entropyImage[y, x, 0]) + int(entropyImage[y, x, 1]) + int(entropyImage[y, x, 2])
            minent, minpathVal = find_min_horz(entropyImage.shape[0],x,y,minImg)
            minPath[y, x] = minpathVal
            minImg[y, x] = minent + pixent

    return minImg, minPath
# end function


def find_min_vert(maxSize, curPlaceX, curPlaceY, minImg):
    values = [999999, 999999, 999999]
    for ctr in range(-1,2):
        if 0 < ctr + curPlaceX < maxSize-1: # changed from 0 <= ctr + curPlaceX < maxSize: otherwise it was always taking image edge
            values[ctr+1] = int(minImg[curPlaceY-1, ctr+curPlaceX])
    return min(values), values.index(min(values)) - 1
# end function


def find_min_horz(maxSize, curPlaceX, curPlaceY, minImg):
    values = [999999, 999999, 999999]
    for ctr in range(-1,2):
        if 0 < ctr + curPlaceY < maxSize-1: # changed from 0 <= ctr + curPlaceX < maxSize: otherwise it was always taking image edge
            values[ctr+1] = int(minImg[ctr+curPlaceY, curPlaceX-1])
    return min(values), values.index(min(values)) - 1
# end function


def remove_vert_seam(image, pix2remove):
    newImg = np.zeros((image.shape[0], image.shape[1]-1, 3),dtype=int)

    for y in range(0,image.shape[0]):
        newX = 0
        for x in range(0, image.shape[1]):
            if pix2remove[y, x] == 0:
                newImg[y, newX] = image[y, x]
                newX += 1
    return newImg
# end function


def remove_horz_seam(image, pix2remove):
    newImg = np.zeros((image.shape[0]-1, image.shape[1], 3),dtype=int)

    for x in range(0,image.shape[1]):
        newY = 0
        for y in range(0, image.shape[0]):
            if pix2remove[y, x] == 0:
                newImg[newY, x] = image[y, x]
                newY += 1
    return newImg
# end function


def get_vert_seam(minVals, minPath):
    path = np.zeros(minPath.shape,dtype=int)
    minSpot = np.argmin(minVals[minVals.shape[0]-1])
    for y in range(minVals.shape[0]-1,-1,-1):
        path[y, minSpot] = 1
        minSpot = minSpot + minPath[y, minSpot]
    return path
# end function


def get_horz_seam(minVals, minPath):
    path = np.zeros(minPath.shape,dtype=int)
    minSpot = np.argmin(minVals[:,minVals.shape[1]-1])
    for x in range(minVals.shape[1]-1,-1,-1):
        path[minSpot, x] = 1
        minSpot = minSpot + minPath[minSpot, x]
    return path
# end function


image = load_img("japan.jpg")

for i in range(0,20):
    print(i)

    entropy = entropy_sobel(image.astype(float)) # Sobel needs array as float
    minValsH, minPathH = find_horz_seam(entropy)
    actMinPathH = get_horz_seam(minValsH, minPathH)
    image = remove_horz_seam(image, actMinPathH)

for i in range(0,20):
    print(i)

    entropy = entropy_sobel(image.astype(float)) # Sobel needs array as float
    minValsV, minPathV = find_vert_seam(entropy)
    actMinPathV = get_vert_seam(minValsV, minPathV)
    image = remove_vert_seam(image, actMinPathV)

cv.imwrite("output.jpg",image)
showim = cv.imread("output.jpg")
cv.imshow("", showim)
cv.waitKey(0)


# f = open("output.txt", "w")
#
#
#
# f.write("Values:" + "\n")
# for y in range(0,minPathV.shape[0]):
#     line = ""
#     for x in range(0,minPathV.shape[1]):
#         showVal = int(entropy[y, x, 0]) + int(entropy[y, x, 1]) + int(entropy[y, x, 2])
#         line += "" + str(showVal) + "\t"
#     f.write(line + "\n")
#
# f.write("\n" + "MinValsV:" + "\n")
# for y in range(0,minPathV.shape[0]):
#     line = ""
#     for x in range(0,minPathV.shape[1]):
#         line += "" + str(minValsV[y, x]) + "\t\t"
#     f.write(line + "\n")
#
# f.write("\n" + "MinPathV:" + "\n")
# for y in range(0,minPathV.shape[0]):
#     line = ""
#     for x in range(0,minPathV.shape[1]):
#         line += "" + str(minPathV[y, x]) + "\t\t"
#     f.write(line + "\n")
#
# f.write("\n" + "MinValsH:" + "\n")
# for y in range(0,minPathH.shape[0]):
#     line = ""
#     for x in range(0,minPathH.shape[1]):
#         line += "" + str(minValsH[y, x]) + "\t\t"
#     f.write(line + "\n")
#
# f.write("\n" + "MinPathH:" + "\n")
# for y in range(0,minPathH.shape[0]):
#     line = ""
#     for x in range(0,minPathH.shape[1]):
#         line += "" + str(minPathH[y, x]) + "\t\t"
#     f.write(line + "\n")
#
# f.write("\n" + "ActualMinPathV:" + "\n")
# for y in range(0,actMinPathV.shape[0]):
#     line = ""
#     for x in range(0,actMinPathV.shape[1]):
#         line += "" + str(actMinPathV[y, x]) + "\t\t"
#     f.write(line + "\n")
#
# f.write("\n" + "ActualMinPathH:" + "\n")
# for y in range(0,actMinPathH.shape[0]):
#     line = ""
#     for x in range(0,actMinPathH.shape[1]):
#         line += "" + str(actMinPathH[y, x]) + "\t\t"
#     f.write(line + "\n")
#
# f.write("\n" + "OrigIm:" + "\n")
# for y in range(0,image.shape[0]):
#     line = ""
#     for x in range(0,image.shape[1]):
#         line += "" + str(image[y, x]) + "\t\t"
#     f.write(line + "\n")
#
# f.write("\n" + "NewIm:" + "\n")
# for y in range(0,newIm.shape[0]):
#     line = ""
#     for x in range(0,newIm.shape[1]):
#         line += "" + str(newIm[y, x]) + "\t\t"
#     f.write(line + "\n")
#
#
# f.close()
