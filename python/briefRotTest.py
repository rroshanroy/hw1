import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy.ndimage as scp
from matplotlib import pyplot as plt
from helper import plotMatches

#Q2.1.6

def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    img1 = cv2.imread('data/cv_cover.jpg')
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    match_dict = {}
    data = []

    for i in range(36):

        # TODO: Rotate Image
        rot_img = scp.rotate(img1, i*10)

        # TODO: Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(img1, rot_img, opts)
    
        # TODO: Update histogram
        match_dict[i*10] = matches.shape[0]
        data.append(matches.shape[0])

        if i==10 or i==20 or i==30:
            plotMatches(img1, rot_img, matches, locs1, locs2, i)



    # TODO: Display histogram
    plt.bar(match_dict.keys(), match_dict.values(), width=2.8)
    plt.xlabel('Rotation (deg)')
    plt.ylabel('Number of matches')
    plt.savefig('pics/rot_bar.png')


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
