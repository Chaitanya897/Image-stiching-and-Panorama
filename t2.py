# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        #print(ipath)
        imgs.append(img)
    "Start you code here"
    img1 = imgs[0]
    for i in range(len(imgs)):
        img2 = imgs[i+1]

        sift = cv2.SIFT_create() 
        (kp1, des1) = sift.detectAndCompute(img1,None)
        (kp2, des2) = sift.detectAndCompute(img2,None)
        #print(des1, des2)

        #sift_img1 = cv2.drawKeypoints(img1, kp1, img1)
        #sift_image2 = cv2.drawKeypoints(img2, kp2, img2)

        #cv2.imshow('image', sift_image2)
        #cv2.imwrite("table-sift.jpg", sift_image2)
        #cv2.waitKey(10000)
        #cv2.destroyAllWindows()

        distance = np.sum((des1[:, np.newaxis, :] - des2[np.newaxis, :, :]) ** 2, axis=-1)
        #print(distance)

        T = 5000

        points_for_image1 = np.where(distance < T)[0]
        image1_matched = np.array([kp1[i].pt for i in points_for_image1])
        #print(image1_matched)

        points_for_image2 = np.where(distance < T)[1]
        image2_matched = np.array([kp2[j].pt for j in points_for_image2])
        #print(image2_matched)
        
        match = np.concatenate((image1_matched, image2_matched), axis = 1)

        #print(match_copy.shape)
        match1 = match[0:47,0:2]
        match2 = match[0:47,2:4]

        H, mask = cv2.findHomography(match1, match2, cv2.RANSAC, 5.0)


        result = cv2.warpPerspective(img1, H, ((img1.shape[1] + img2.shape[1])*1, (img2.shape[0] + img1.shape[0])*1))
        result[0:img2.shape[0],0:img2.shape[1]] = img2
        cv2.imshow('image', result)
        cv2.imwrite(savepath, result)

        img1 = result

    #return overlap_arr
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    '''with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)'''
