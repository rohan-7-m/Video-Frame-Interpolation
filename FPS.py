# This script takes in an [x] FPS video and outputs 2*[x] FPS video by applying results
#  of frame interpolation using FI-CNN.


import sys
import os
import time
import random
import numpy as np
import cv2
from FI_unet import UNet_Model


def Video_loader(video_path):
    Video = cv2.VideoCapture(video_path)
    num_frames = int(Video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = Video.get(cv2.CAP_PROP_FPS)

    vid_arr = np.zeros(shape=(num_frames, 128, 384, 3), dtype="uint8")
    for i in range(num_frames):
        ret, frame = Video.read()
        if frame is None:
            continue
        vid_arr[i] = cv2.resize(frame, (384,128))
    return vid_arr, fps



def double_vid_fps(vid_arr):

    model = UNet_Model((128, 384, 6))
    model.load_weights("./../model_weights/weights_unet2_finetune_youtube_100epochs.hdf5")

    # new_vid_arr = np.zeros(shape=(len(vid_arr)*2, 128, 384, 3))
    new_vid_arr = []
    new_vid_arr.append(vid_arr[0])
    for i in range(1, len(vid_arr)):
        # if i % (len(vid_arr) / 10) == 0:
        #     print ("FPS doubling is {0}% done.".format((i / (len(vid_arr) / 10) * 10)))


        pred = model.predict(np.expand_dims(np.transpose(np.concatenate((vid_arr[i-1], vid_arr[i]), axis=2)/255., (2, 0, 1)), axis=0))
        new_vid_arr.append((np.transpose(pred[0], (1, 2, 0))*255).astype("uint8"))
        new_vid_arr.append(vid_arr[i])

    return np.asarray(new_vid_arr)

def save_vid(vid_arr, vid_out_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(vid_out_path, fourcc, fps, (384, 128))

    for i in range(len(vid_arr)):
        out.write(vid_arr[i])

def main():

    # vid_dir = "/Users/rohan.m/Desktop/fps"
    # vid_fn = "30sec_nature.mp4"
    vid_file = "Bleach - 300 [720p] [Dual] @Anime_Gallery.mp4"
    Directory = "./../results/videos/"

    vid_arr, fps = Video_loader(os.path.join(Directory, vid_file))

    double_vid_arr = double_vid_fps(vid_arr)

    save_vid(double_vid_arr, Directory + vid_file.split('.')[0] + "_double_60.avi", fps=fps*2)


if __name__ == '__main__':
    model = UNet_Model((128, 384, 6))
    main()
