import os
import cv2
import h5py
import numpy as np
from behaviourPipeline.data import process_h5py_data, bsoid_format
from behaviourPipeline.preprocessing import likelihood_filter, trim_data
from behaviourPipeline.features import extract_comb_feats, aggregate_features

import logging
logger = logging.getLogger(__name__)

def trim_video_data(data, fps):
    return trim_data(data, fps)

def extract_data_from_video(video_file, bodyparts, fps, conf_threshold, filter_thresh):
    # get keypoint data from video file
    conf, pos = process_h5py_data(h5py.File(video_file, 'r'))
    bsoid_data = bsoid_format(conf, pos)
    fdata, perc_filt = likelihood_filter(bsoid_data, conf_threshold, bodyparts)

    if perc_filt > filter_thresh: logger.warning(f"% data filtered from {os.path.split(video_file)[-1]} too high ({perc_filt}%)")
    for key, data in fdata.items():
        fdata[key] = trim_video_data(data, fps)
    
    return fdata

def video_frame_predictions(video_file, clf, stride_window, bodyparts, fps, conf_threshold, filter_thresh):
    fdata = extract_data_from_video(video_file, bodyparts, fps, conf_threshold, filter_thresh)
    geom_feats = extract_comb_feats(fdata, fps)
    fs_feats = [aggregate_features(geom_feats[i:], stride_window) for i in range(stride_window)]
    fs_labels = [clf.predict(f).squeeze() for f in fs_feats]
    max_len = max([f.shape[0] for f in fs_labels])
    for i, f in enumerate(fs_labels):
        pad_arr = -1 * np.ones((max_len,))
        pad_arr[:f.shape[0]] = f
        fs_labels[i] = pad_arr
    labels = np.array(fs_labels).flatten('F')
    labels = labels[labels >= 0]
    return labels

def videomaker(frames, fps, outfile):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    height, width, _ = frames[0].shape

    video = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
    for i in range(len(frames)):
        video.write(frames[i].astype(np.uint8))

    cv2.destroyAllWindows()
    video.release()

def add_group_label_to_frame(frames, label):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.6
    # parameters for adding text
    text = f'Group {label}'
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    text_offset_x = 20
    text_offset_y = 20
    box_coords = ((text_offset_x - 12, text_offset_y + 12), (text_offset_x + text_width + 12, text_offset_y - text_height - 8))

    for i, frame in enumerate(frames):
        frames[i] = cv2.rectangle(frame, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
        frames[i] = cv2.putText(frame, text, (text_offset_x, text_offset_y), font,
                            fontScale=font_scale, color=(255, 255, 255), thickness=1)
    
    return frames

def behaviour_clips(video_dir, min_bout_len, n_examples, outdir):
    