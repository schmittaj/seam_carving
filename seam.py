import cv2 as cv
import numpy as np


global entropyFunc
global win
global frame
global mask
global image
global topLeft
global bottomRight


def load_img(imgName):
    img = cv.imread(imgName,cv.IMREAD_COLOR)
    return img
# end function


def entropy_saliency(image):
    saliency = cv.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image.astype(np.uint8))
    saliencyMap = (saliencyMap * 255).astype("uint8")
    return saliencyMap
# end function


def entropy_sobel(image):
    img = cv.Sobel(image.astype(float), -1, 1, 1)
    return img
# end function


def entropy_scharr(image):
    imgx = cv.Scharr(image.astype(float), cv.CV_64F, 1, 0)
    imgy = cv.Scharr(image.astype(float), cv.CV_64F, 0, 1)
    img = cv.addWeighted(imgx, 0.5, imgy, 0.5, 0)
    return img
# end function


def entropy_laplacian(image):
    lap = cv.Laplacian(image.astype(float), cv.CV_64F)
    return lap
# end function


def find_vert_seam(entropyImage):
    minImg = np.zeros((entropyImage.shape[0], entropyImage.shape[1]), dtype=int)
    minPath = np.zeros((entropyImage.shape[0], entropyImage.shape[1]), dtype=int)
    for x in range(0, entropyImage.shape[1]):
        if entropyImage[0, x].shape == ():
            minImg[0, x] = entropyImage[0, x]
        else:
            minImg[0,x] = entropyImage[0, x, 0] + entropyImage[0, x, 1] + entropyImage[0, x, 2]

    # going top to bottom
    for y in range(1, entropyImage.shape[0]):
        for x in range(0,entropyImage.shape[1]):
            if entropyImage[y, x].shape == ():
                pixent = entropyImage[y, x]
            else:
                pixent = int(entropyImage[y, x, 0]) + int(entropyImage[y, x, 1]) + int(entropyImage[y, x, 2])

            minent, minpathVal = find_min_vert(entropyImage.shape[1],x,y,minImg)
            minPath[y, x] = minpathVal
            minImg[y, x] = minent + pixent

    return minImg, minPath
# end function


def apply_entropy_mask(entropyImage, mask):
    newEnt = np.zeros((entropyImage.shape[0], entropyImage.shape[1]), dtype=int)
    for y in range(0, entropyImage.shape[0]):
        for x in range(0, entropyImage.shape[1]):
            if entropyImage[y, x].shape == ():
                if mask[y, x] == 1:
                    newEnt[y, x] = 255
                elif mask[y, x] == -1:
                    newEnt[y, x] = 0
                else:
                    newEnt[y, x] = entropyImage[y, x]
            else:
                if mask[y, x] == 1:
                    newEnt[y, x, 0] = 255
                    newEnt[y, x, 1] = 255
                    newEnt[y, x, 2] = 255
                elif mask[y, x] == -1:
                    newEnt[y, x, 0] = 0
                    newEnt[y, x, 1] = 0
                    newEnt[y, x, 2] = 0
                else:
                    newEnt[y, x, 0] = entropyImage[y, x, 0]
                    newEnt[y, x, 1] = entropyImage[y, x, 1]
                    newEnt[y, x, 2] = entropyImage[y, x, 2]
    return newEnt
# end function


def find_horz_seam(entropyImage):
    minImg = np.zeros((entropyImage.shape[0], entropyImage.shape[1]), dtype=int)
    minPath = np.zeros((entropyImage.shape[0], entropyImage.shape[1]), dtype=int)
    for y in range(0, entropyImage.shape[0]):
        if entropyImage[y,0].shape == ():
            minImg[y,0] = entropyImage[y,0]
        else:
            minImg[y,0] = entropyImage[y, 0, 0] + entropyImage[y, 0, 1] + entropyImage[y, 0, 2]

    # going left to right
    for x in range(1, entropyImage.shape[1]):
        for y in range(0,entropyImage.shape[0]):
            if entropyImage[y, x].shape == ():
                pixent = entropyImage[y, x]
            else:
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

def remove_vert_seam_mask(image, pix2remove):
    newImg = np.zeros((image.shape[0], image.shape[1]-1),dtype=int)

    for y in range(0,image.shape[0]):
        newX = 0
        for x in range(0, image.shape[1]):
            if pix2remove[y, x] == 0:
                newImg[y, newX] = image[y, x]
                newX += 1
    return newImg
# end function


def remove_horz_seam_mask(image, pix2remove):
    newImg = np.zeros((image.shape[0]-1, image.shape[1]),dtype=int)

    for x in range(0,image.shape[1]):
        newY = 0
        for y in range(0, image.shape[0]):
            if pix2remove[y, x] == 0:
                newImg[newY, x] = image[y, x]
                newY += 1
    return newImg
# end function


def add_vert_seam(image, pix2add):
    newImg = np.zeros((image.shape[0], image.shape[1]+1, 3),dtype=int)

    for y in range(0,newImg.shape[0]):
        oldX = 0
        found = False
        for x in range(0, newImg.shape[1]):
            pixVal = image[y, oldX]
            oldX += 1
            if pix2add[y, oldX-1] == 1 and not found:
                pixVal = calcAvgPixVal(y, oldX, image, image.shape[0]-1, image.shape[1]-1)
                oldX -= 1
                found = True
            newImg[y, x] = pixVal
    return newImg
# end function


def add_horz_seam(image, pix2add):
    newImg = np.zeros((image.shape[0]+1, image.shape[1], 3),dtype=int)

    for x in range(0,newImg.shape[1]):
        oldY = 0
        found = False
        for y in range(0, newImg.shape[0]):
            pixVal = image[oldY, x]
            oldY += 1
            if pix2add[oldY-1, x] == 1 and not found:
                pixVal = calcAvgPixVal(oldY, x, image, image.shape[0]-1, image.shape[1]-1)
                oldY -= 1
                found = True
            newImg[y, x] = pixVal
    return newImg
# end function


def calcAvgPixVal(y, x, img, ymax, xmax):
    totals = np.zeros(3,dtype=int)
    count = 0;
    if x != 0:
        if y != 0:
            totals[0] = totals[0] + img[y-1, x-1, 0]
            totals[1] = totals[1] + img[y-1, x-1, 1]
            totals[2] = totals[2] + img[y-1, x-1, 2]
            count += 1
            totals[0] = totals[0] + img[y-1, x, 0]
            totals[1] = totals[1] + img[y-1, x, 1]
            totals[2] = totals[2] + img[y-1, x, 2]
            count += 1
        totals[0] = totals[0] + img[y, x-1, 0]
        totals[1] = totals[1] + img[y, x-1, 1]
        totals[2] = totals[2] + img[y, x-1, 2]
        count += 1
        if y != ymax:
            totals[0] = totals[0] + img[y+1, x-1, 0]
            totals[1] = totals[1] + img[y+1, x-1, 1]
            totals[2] = totals[2] + img[y+1, x-1, 2]
            count += 1
    if x != xmax:
        if y != 0:
            totals[0] = totals[0] + img[y-1, x+1, 0]
            totals[1] = totals[1] + img[y-1, x+1, 1]
            totals[2] = totals[2] + img[y-1, x+1, 2]
            count += 1
        totals[0] = totals[0] + img[y, x+1, 0]
        totals[1] = totals[1] + img[y, x+1, 1]
        totals[2] = totals[2] + img[y, x+1, 2]
        count += 1
        if y != ymax:
            totals[0] = totals[0] + img[y+1, x+1, 0]
            totals[1] = totals[1] + img[y+1, x+1, 1]
            totals[2] = totals[2] + img[y+1, x+1, 2]
            count += 1
            totals[0] = totals[0] + img[y+1, x, 0]
            totals[1] = totals[1] + img[y+1, x, 1]
            totals[2] = totals[2] + img[y+1, x, 2]
            count += 1
    totals[0] = totals[0] / count;
    totals[1] = totals[1] / count;
    totals[2] = totals[2] / count;

    #totals = totals.astype(int)
    return totals
#end function


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


def perform_removal(entropy_function, image, linesToRemove):  # alternates removing vertical and horizontal
    img = np.copy(image)
    for i in range(0,linesToRemove):
        print(i)

        if i%2 == 0:
            entropy = entropy_function(img)
            minValsH, minPathH = find_horz_seam(entropy)
            actMinPathH = get_horz_seam(minValsH, minPathH)
            img = remove_horz_seam(img, actMinPathH)
        else:
            entropy = entropy_function(img)
            minValsV, minPathV = find_vert_seam(entropy)
            actMinPathV = get_vert_seam(minValsV, minPathV)
            img = remove_vert_seam(img, actMinPathV)
    return img
# end function


def perform_addition(entropy_function, image, linesToAdd):  # alternates adding vertical and horizontal
    img = np.copy(image)
    for i in range(0,linesToAdd):
        print(i)

        if i%2 == 0:
            entropy = entropy_function(img)
            minValsH, minPathH = find_horz_seam(entropy)
            actMinPathH = get_horz_seam(minValsH, minPathH)
            img = add_horz_seam(img, actMinPathH)
        else:
            entropy = entropy_function(img)
            minValsV, minPathV = find_vert_seam(entropy)
            actMinPathV = get_vert_seam(minValsV, minPathV)
            img = add_vert_seam(img, actMinPathV)
    return img
# end function


def perform_addition_horz(entropy_function, image, linesToAdd):
    img = np.copy(image)
    for i in range(0,linesToAdd):
        entropy = entropy_function(img)
        minValsH, minPathH = find_horz_seam(entropy)
        actMinPathH = get_horz_seam(minValsH, minPathH)
        img = add_horz_seam(img, actMinPathH)
    return img
# end function


def perform_addition_vert(entropy_function, image, linesToAdd):
    img = np.copy(image)
    for i in range(0,linesToAdd):
        print(i)
        entropy = entropy_function(img)
        minValsV, minPathV = find_vert_seam(entropy)
        actMinPathV = get_vert_seam(minValsV, minPathV)
        img = add_vert_seam(img, actMinPathV)
    return img
# end function


def perform_removal_horz(entropy_function, image, linesToRemove):
    img = np.copy(image)
    for i in range(0,linesToRemove):
        print(i)
        entropy = entropy_function(img)
        minValsH, minPathH = find_horz_seam(entropy)
        actMinPathH = get_horz_seam(minValsH, minPathH)
        img = remove_horz_seam(img, actMinPathH)
    return img
# end function


def perform_removal_vert(entropy_function, image, linesToRemove):
    img = np.copy(image)
    for i in range(0,linesToRemove):
        print(i)
        entropy = entropy_function(img)
        minValsV, minPathV = find_vert_seam(entropy)
        actMinPathV = get_vert_seam(minValsV, minPathV)
        img = remove_vert_seam(img, actMinPathV)
    return img
# end function


def perform_removal_horz_with_mask(entropy_function, mask, image, linesToRemove):
    img = np.copy(image)
    msk = np.copy(mask)
    for i in range(0,linesToRemove):
        print(i)
        entropy = entropy_function(img)
        entropy = apply_entropy_mask(entropy, msk)
        minValsH, minPathH = find_horz_seam(entropy)
        actMinPathH = get_horz_seam(minValsH, minPathH)
        msk = remove_horz_seam_mask(msk, actMinPathH)
        img = remove_horz_seam(img, actMinPathH)
    return img, msk
# end function


def perform_removal_vert_with_mask(entropy_function, mask, image, linesToRemove):
    img = np.copy(image)
    msk = np.copy(mask)
    for i in range(0,linesToRemove):
        print(i)
        entropy = entropy_function(img)
        entropy = apply_entropy_mask(entropy, msk)
        minValsV, minPathV = find_vert_seam(entropy)
        actMinPathV = get_vert_seam(minValsV, minPathV)
        msk = remove_vert_seam_mask(msk, actMinPathV)
        img = remove_vert_seam(img, actMinPathV)
    return img, msk
# end function


def bound(event, x, y, flags, param):

    global topLeft
    global bottomRight

    done = False
    if event == cv.EVENT_LBUTTONDOWN:
        topLeft = [x, y]
    elif event == cv.EVENT_LBUTTONUP:
        bottomRight = [x, y]
        done = True

    if(done):
        global mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=int)
        for a in range(topLeft[1],bottomRight[1]):
            for b in range(topLeft[0],bottomRight[0]):
                mask[a, b] = -1
        print((bottomRight[0]-topLeft[0]))
        out, msk = perform_removal_vert_with_mask(entropy_saliency, mask, image, (bottomRight[0]-topLeft[0]))
        cv.imwrite("maskOut.jpg", out)
        image2 = load_img("maskOut.jpg")
        cv.imshow('', image2)
        print("Done")
# end function


def load_image():
    filename = askopenfilename()
    global image
    image = load_img(filename)
    cv.imshow('', image)
    cv.setMouseCallback('', bound)
# end function


# while 1:
#     win.update_idletasks()
#     win.update()

# image = load_img("test2.jpg")
# ent = entropy_saliency(image)
# mask = np.array([[0,0,0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,-1,-1,0,0],
#                  [0,0,0,0,0,0,-1,-1,0,0],
#                  [0,0,0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0,0,0],
#                  [0,0,0,0,0,0,0,0,0,0]])
# newEnt = apply_entropy_mask(ent, mask)
# image, mask = perform_removal_vert_with_mask(entropy_saliency,mask,image,2)
# image, mask = perform_removal_horz_with_mask(entropy_saliency,mask,image,2)

#cv.imwrite("output.jpg",image)

image = load_img("dragon.jpg")
image = perform_addition(saliency, image, 40)
#image = perform_removal(saliency, image, 40)
cv.imwrite("dragoadd.jpg",image)

# there's a bug with opencv that isn't letting us show the new image directly
#showim = cv.imread("output.jpg")  # so have to write to a file and read it back in to show
#cv.imshow("", showim)
#cv.waitKey(0)



# Text file output for debugging
# f = open("output.txt", "w")
#
#
#
# f.write("Values:" + "\n")
# for y in range(0,minPathV.shape[0]):
#     line = ""
#     for x in range(0,minPathV.shape[1]):
#         #showVal = int(entropy[y, x, 0]) + int(entropy[y, x, 1]) + int(entropy[y, x, 2])
#         showVal = int(entropy[y, x])
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
# # f.write("\n" + "MinValsH:" + "\n")
# # for y in range(0,minPathH.shape[0]):
# #     line = ""
# #     for x in range(0,minPathH.shape[1]):
# #         line += "" + str(minValsH[y, x]) + "\t\t"
# #     f.write(line + "\n")
# #
# # f.write("\n" + "MinPathH:" + "\n")
# # for y in range(0,minPathH.shape[0]):
# #     line = ""
# #     for x in range(0,minPathH.shape[1]):
# #         line += "" + str(minPathH[y, x]) + "\t\t"
# #     f.write(line + "\n")
#
# f.write("\n" + "ActualMinPathV:" + "\n")
# for y in range(0,actMinPathV.shape[0]):
#     line = ""
#     for x in range(0,actMinPathV.shape[1]):
#         line += "" + str(actMinPathV[y, x]) + "\t\t"
#     f.write(line + "\n")
#
# # f.write("\n" + "ActualMinPathH:" + "\n")
# # for y in range(0,actMinPathH.shape[0]):
# #     line = ""
# #     for x in range(0,actMinPathH.shape[1]):
# #         line += "" + str(actMinPathH[y, x]) + "\t\t"
# #     f.write(line + "\n")
#
# f.write("\n" + "OrigIm:" + "\n")
# for y in range(0,ogimage.shape[0]):
#     line = ""
#     for x in range(0,ogimage.shape[1]):
#         line += "" + str(ogimage[y, x]) + "\t\t"
#     f.write(line + "\n")
#
# f.write("\n" + "NewIm:" + "\n")
# for y in range(0,image.shape[0]):
#     line = ""
#     for x in range(0,image.shape[1]):
#         line += "" + str(image[y, x]) + "\t\t"
#     f.write(line + "\n")
#
#
# f.close()
