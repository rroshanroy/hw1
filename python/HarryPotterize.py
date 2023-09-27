import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from helper import plotMatches

# Import necessary functions

# Q2.2.4

def warpImage(start, target, transform, opts):

    matches, locs1, locs2 = matchPics(start, target, opts)
    
    # flipping the dimensions as per FAQ
    # code taken from: https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
    locs1[:, 0], locs1[:, 1] = locs1[:, 1], locs1[:, 0].copy()
    locs2[:, 0], locs2[:, 1] = locs2[:, 1], locs2[:, 0].copy()
    
    H, _ = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]], opts)
    
    comp_img = compositeH(np.linalg.inv(H), cv2.resize(transform, (start.shape[1], start.shape[0])), target)

    return comp_img
    

if __name__ == "__main__":

    opts = get_opts()

    cv_cov = cv2.imread('data/cv_cover.jpg')
    cv_desk = cv2.imread('data/cv_desk.png')
    hp_cov = cv2.imread('data/hp_cover.jpg')

    save_path = f'pics/hpdesk_maxiter{opts.max_iters}_tol{opts.inlier_tol}haha2.jpg'

    warp_image = warpImage(cv_cov, cv_desk, hp_cov, opts)
    cv2.imwrite(save_path, warp_image)


