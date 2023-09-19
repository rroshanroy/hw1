import numpy as np
import cv2
from opts import get_opts
import random


def computeH(p1, p2):
    #Q2.2.1
    # TODO: Compute the homography between two sets of points

    rows = 2 * p1.shape[0]
    A = np.zeros((rows,9))

    x1 = p1[:, 0]
    x2 = p2[:, 0]

    y1 = p1[:, 1]
    y2 = p2[:, 1]

    A[::2, 0] = x2
    A[::2, 1] = y2
    A[::2, 2] = 1
    A[1::2, 3] = x2
    A[1::2, 4] = y2
    A[1::2, 5] = 1
    A[::2, 6] = -x2 * x1
    A[1::2, 6] = -x2 * y1
    A[::2, 7] = -y2 * x1
    A[1::2, 7] = -y2 * y1
    A[::2, 8] = -x1
    A[1::2, 8] = -y1
    
    _, _, Vh = np.linalg.svd(A)
    H2to1 = Vh[-1, :].reshape(3,3)  # grab first row of V^T matrix, corresponds to singular value of 0


    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    # TODO: Compute the centroid of the points
    x1_c = np.mean(x1, axis=0).astype('int64')
    x2_c = np.mean(x2, axis=0).astype('int64')


    # TODO: Shift the origin of the points to the centroid
    x1 -= x1_c
    x2 -= x2_c

    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_max = np.max(np.linalg.norm(x1, axis=1))
    x2_max = np.max(np.linalg.norm(x1, axis=1))

    x1_sc = np.sqrt(2) / x1_max
    x2_sc = np.sqrt(2) / x2_max

    x1 = x1 * x1_sc 
    x2 = x2 * x2_sc

    # TODO: Similarity transform 1
    T1 = np.array(
        [[x1_sc, 0, -x1_sc*x1_c[0]],
        [0, x1_sc, -x1_sc*x1_c[1]],
        [0, 0, 1]]
    )

    # TODO: Similarity transform 2
    T2 = np.array(
        [[x2_sc, 0, -x2_sc*x2_c[0]],
        [0, x2_sc, -x2_sc*x2_c[1]],
        [0, 0, 1]]
    )

    # TODO: Compute homography
    H_norm = computeH(x1, x2)


    # TODO: Denormalization
    H2to1 = np.matmul(np.linalg.inv(T1), np.matmul(H_norm, T2))

    return H2to1/H2to1[2,2]




def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    all_inds = [i for i in range(len(locs1))]
    best_count = 0
    inliers = None
    bestH2to1 = None

    for iter in range(max_iters):
        # choose four sets of points from the matched arrays
        inds = np.array(random.sample(all_inds, 4))
        # inds = inds_hardcoded = np.array([1,2,3,4])
        mask = np.zeros(len(locs1), dtype=bool)
        mask[inds] = True

        H = computeH_norm(locs1[inds], locs2[inds])
        # H2, _ = cv2.findHomography(locs2[inds], locs1[inds])

        # project the points not used to compute homography
        hom_in = np.column_stack((locs2[~mask], np.ones(locs2[~mask].shape[0])))
        # hom_in = np.column_stack((locs2, np.ones(locs2.shape[0])))

        # projected_x2tox1 = np.matmul(hom_in, H)
        # projected_x2tox1 = projected_x2tox1[:2, :] / projected_x2tox1[2, :][np.newaxis, :]
        hom_out = np.matmul(H, hom_in.T).T  # (3,14)
        projected_x2tox1 = hom_out[:, :2] / hom_out[:, 2][:, np.newaxis]
        
        truth_x1 = locs1[~mask]
        inlier_inds = np.linalg.norm(truth_x1 - projected_x2tox1, axis=1) <= inlier_tol

        if np.sum(inlier_inds) > best_count:
            best_count = np.sum(inlier_inds)
            inliers = np.copy(inlier_inds)
            bestH2to1 = np.copy(H)

        
    return bestH2to1, inliers


def compositeH(H2to1, template, img, large_canvas=False):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.

    canvas = (img.shape[1], 2*img.shape[0]) if large_canvas else (img.shape[1], img.shape[0])

    # TODO: Create mask of same size as template
    mask = np.ones_like(template)

    # TODO: Warp mask by appropriate homography
    composite_mask = cv2.warpPerspective(mask, H2to1, canvas)

    # TODO: Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H2to1, canvas)

    # TODO: Use mask to combine the warped template and the image
    composite_img = np.where(composite_mask, warped_template, img)
    
    return composite_img

if __name__=='__main__':
    pt1 = np.random.rand(4,2)
    pt2 = np.random.rand(4,2)
    

    H = computeH(pt1, pt2)
    H2, _ = cv2.findHomography(pt2, pt1)
    H3 = computeH_norm(pt1, pt2)

    opts = get_opts()
    # bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)
    print("Done")

    


