import os
import h5py
from behaviourPipeline.data import process_h5py_data, bsoid_format

def extract_data_from_video(video_file):
    # get keypoint data from video file
    conf, pos = process_h5py_data(h5py.File(video_file, 'r'))
    bsoid_data = bsoid_format(conf, pos)