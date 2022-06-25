#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    #image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #print(image1)
    #image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #print(image2)

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
    
    #Threshold
    T = 3000

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


    result = cv2.warpPerspective(img1, H, (2*img1.shape[1] + img2.shape[1], img2.shape[0] + img1.shape[0]))

    

    result[0:img2.shape[0],0:img2.shape[1]] = img2

    cv2.imshow('image', result)
    cv2.imwrite(savepath, result) 




    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)   
