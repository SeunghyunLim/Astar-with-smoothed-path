import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import pyfmm
import time
import random



def convert2list(img):
    height, width = img.shape
    maze = np.zeros((height, width), np.uint8)
    for i in range(width):
        for j in range(height):
            maze[j][i] = 1 if img[j][i] > 0 else 0

    return maze.tolist()

def img2binList(img, lenWidth, GRID_SIZE=50, verbose=0):
    global DISTANCECOSTMAP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, unreversed_gray = cv2.threshold(gray, 112, 255, cv2.THRESH_BINARY)
    _, gray = cv2.threshold(gray, 112, 255, cv2.THRESH_BINARY_INV)
    if verbose:
        cv2.imshow("img", gray)
        cv2.waitKey(0)

    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    locs = []

    height, width = gray.shape
    tmp = np.zeros((height, width), np.uint8)

    idxLargest = 0
    areaLargest = 0
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        if w * h > areaLargest:
            idxLargest = i
            areaLargest = w * h
        cv2.rectangle(tmp, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if verbose:
        # print("found largest contour outline")
        cv2.imshow("img", tmp)
        cv2.waitKey(0)

    # print("cropping image as largest contour")
    (x, y, w, h) = cv2.boundingRect(cnts[idxLargest])
    gray = gray[y:y + h, x:x + w]
    unreversed_gray = unreversed_gray[y:y + h, x:x + w]
    if verbose:
        cv2.imshow("img", cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(0)
    if verbose:
        print("Ya")
        cv2.imshow("img", cv2.resize(unreversed_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(0)
    global mapWidth
    global mapHeight
    mapWidth = (int)(lenWidth // GRID_SIZE)
    mapHeight = (int)((h / w) * lenWidth // GRID_SIZE)
    print("the map will be created by the size: " + str(mapWidth) + " X " + str(mapHeight))
    resized_gray = imutils.resize(gray, width=mapWidth)  # resize the map for convolution
    resized_unreversed_gray = imutils.resize(unreversed_gray, width=mapWidth)
    _, resized_gray = cv2.threshold(resized_gray, 1, 255, cv2.THRESH_BINARY)
    _, resized_unreversed_gray = cv2.threshold(resized_unreversed_gray, 0, 255, cv2.THRESH_BINARY)
    if verbose:
        cv2.imshow("img", resized_gray)
        cv2.waitKey(0)
    if verbose:
        print("way")
        cv2.imshow("img", resized_unreversed_gray)
        cv2.waitKey(0)
    maze = convert2list(resized_gray)
    reversed_maze = convert2list(resized_unreversed_gray)
    my_maze = np.array(maze)
    solution = pyfmm.march(my_maze == 1, batch_size=10000)[0] # NOTE : white area means walkable area
    DISTANCECOSTMAP = solution

    # cv2.destroyAllWindows()
    return maze

def walkable_area_contour(maze, x_real, y_real, verbose=0):
    maze = np.array(maze).astype(np.uint8)
    maze *= 255
    maze = cv2.resize(maze, None, fx=7, fy=7, interpolation=cv2.INTER_NEAREST)
    _, reversed_maze = cv2.threshold(maze, 112, 255, cv2.THRESH_BINARY_INV)
    contoured_maze = reversed_maze
    contours, hierarchy = cv2.findContours(reversed_maze, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    idxLargest = 0
    areaLargest = 0
    contour_list = []
    # loop over the contours

    for (i, c) in enumerate(contours):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        contour_list.append([i, w * h])
    print(contour_list)

    for (i, c) in enumerate(contours):
        if cv2.pointPolygonTest(contours[contour_list[i][0]], (x_real * 7, y_real * 7), False) == 1:
            idxLargest = i
            reference_idx = i-1
            print(idxLargest, reference_idx)
            cv2.drawContours(contoured_maze, [contours[idxLargest]], 0, (112, 0, 0), 3)
            break
    # cv2.drawContours(contoured_maze, [contours[idxLargest]], 0, (112, 0, 0), 3)  # blue

    if idxLargest == 0:
        reference_idx = 0

    cv2.drawContours(contoured_maze, [contours[idxLargest]], 0, (112, 0, 0), 3)
    cv2.drawContours(contoured_maze, [contours[reference_idx]], 0, (112, 0, 0), 3)

    if verbose:
        cv2.imshow("contoured", contoured_maze)
        cv2.waitKey(0)

    return (contours[idxLargest], contours[reference_idx], idxLargest, reference_idx)

def curiosityEngine(area, x_range, y_range, verbose=0):
    starting_time = time.time()
    contour = area[0]
    reference_contour = area[1]
    idxLargest = area[2]
    while True:
        x = random.randrange(x_range)
        y = random.randrange(y_range)
        #print(cv2.pointPolygonTest(contour, (x * 7, y * 7), False))]
        if idxLargest == 0:
            if cv2.pointPolygonTest(contour, (x * 7, y * 7), False) == 1:
                return (y, x)
                break
        if cv2.pointPolygonTest(contour, (x * 7, y * 7), False) == 1:
            if cv2.pointPolygonTest(reference_contour, (x * 7, y * 7), False) == -1:
                return (y, x)
                break
    if verbose:
        print("Curiosity Engine Calculation Time :", time.time() - starttime)

if __name__ == '__main__':
    img = cv2.imread("E5_223.jpg")
    # img = cv2.imread("wtest.PNG")
    cv2.imshow('Sample A* algorithm run with distance cost', img)
    cv2.waitKey(0)
    starttime = time.time()
    maze = img2binList(img, lenWidth=500.0, GRID_SIZE=5, verbose=0)  # all unit is cm
    x_real_initial = 20
    y_real_initial = 20
    area = walkable_area_contour(maze, x_real_initial, y_real_initial, verbose=1)
    print("time :", time.time() - starttime)
    while True:
        # start = (7, 7)
        # start = (10, 25)
        start = (x_real_initial, y_real_initial)
        end = curiosityEngine(area, mapWidth, mapHeight)
        print("Start = ", start, "and End = ", end)
        showmaze = np.array(maze).astype(np.uint8)
        showmaze *= 255
        showmaze[start[0]][start[1]] = 150
        showmaze[end[0]][end[1]] = 150
        showmaze = cv2.resize(showmaze, None, fx=7, fy=7, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Walkable Area Extraction', showmaze)
        cv2.waitKey(0)