import numpy as np
import cv2
from HarryPotterize import warpImage
from opts import get_opts
import time

#Import necessary functions

from helper import loadVid


#Write script for Q3.1
if __name__=='__main__':
    ar_frames = loadVid('data/ar_source.mov')
    bk_frames = loadVid('data/book.mov')
    cv_cover = cv2.imread('data/cv_cover.jpg')

    save_dir = 'ar_warp'

    opts = get_opts()
    warped_frames = []

    prev_time = time.time()
    for i in range(min(ar_frames.shape[0], bk_frames.shape[0])):
    #for i in range(10):
        bk_frame = bk_frames[i]
        ar_frame = ar_frames[i]

        # crop the middle of the ar_frame
        y,x,_ = ar_frame.shape
        crop_x = x//3
        crop_y = y//3
        start_x = x//2-(crop_x //2)
        start_y = y//2-(crop_y //2) 
        ar_crop = ar_frame[start_y:start_y+crop_y, start_x:start_x+crop_x, :]

        try:
            warp_ar = warpImage(cv_cover, bk_frame, ar_crop, opts)
        except:
            print(f"Warning: Failed on frame {i+1}")
            continue

        warped_frames.append(warp_ar)
        #cv2.imwrite(f'ar_warp/frame{i}_img.jpg', warp_ar)
        # cv2.imwrite(f'{save_dir}/frame_{i+1}.jpg', warp_ar)

        if i%10==0 :
            cur_time = time.time()
            print(f"On frame {i+1}. Runtime: {cur_time-prev_time}s per 10 frames")
            prev_time = cur_time

    out = cv2.VideoWriter("ar_warp/final_warped.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (bk_frames[0].shape[1], bk_frames[0].shape[0]))
    for i, frame in enumerate(warped_frames):
        out.write(frame) # frame is a numpy.ndarray with shape (1280, 720, 3)
        #cv2.imwrite(f'ar_warp/test{i}_img.jpg', frame)
    out.release()

    print("Done")