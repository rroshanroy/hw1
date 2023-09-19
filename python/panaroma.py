import numpy as np
import cv2
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

# Import necessary functions




# Q4
if __name__=='__main__':
    opts = get_opts()
    
    pan_left = cv2.imread('data/pano_left.jpg')
    pan_right = cv2.imread('data/pano_right.jpg')
    
    aug_pan_right = np.column_stack((np.zeros_like(pan_right), pan_right))
    matches, locs1, locs2 = matchPics(aug_pan_right, pan_left, opts)
    # flipping the dimensions as per FAQ
    locs1[:, 0], locs1[:, 1] = locs1[:, 1], locs1[:, 0].copy()
    locs2[:, 0], locs2[:, 1] = locs2[:, 1], locs2[:, 0].copy()

    H, _ = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]], opts)

    comp_img = compositeH(H, pan_left, aug_pan_right)#, large_canvas=True)
    # cv2.imwrite('pics/pano_left2right_nocrop.jpg', comp_img)

    gray_comp_img = cv2.cvtColor(comp_img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray_comp_img,1,255,cv2.THRESH_BINARY)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    x,y,w,h = cv2.boundingRect(cnt)
    max_x = np.max(approx[:, 0, 0])
    max_y = np.max(approx[:, 0, 1])
    min_x = np.min(approx[:, 0, 0])
    min_y = np.min(approx[:, 0, 1])

    # crop = comp_img[y:y+h,x:x+w]
    crop = comp_img[min_y:max_y,min_x:max_x]
    cv2.imwrite('pics/pano_left2right_crop2.jpg',crop)

    
    print("Done")