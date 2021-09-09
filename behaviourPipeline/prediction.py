import os
import cv2
import h5py
import ffmpeg
import numpy as np
from collections import namedtuple

from behaviourPipeline.data import process_h5py_data, bsoid_format
from behaviourPipeline.preprocessing import likelihood_filter
from behaviourPipeline.features import extract_comb_feats, aggregate_features

import logging
logger = logging.getLogger(__name__)

def trim_video_data(data, fps, isvideo=False):
    # remove first and last 5 mins
    end_trim = 5 * 60 * fps
    if isvideo:
        success, _ = data.read()
        for _ in range(end_trim):
            if not success: break
            success, _ = data.read()
        return data
        
    data = data[end_trim:-end_trim]
    
    # consider only first 50 mins
    clip = 50 * 60 * fps
    data = data[:clip+1]
    
    return data
        
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
    if not frames: return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

def bouts_from_video(behaviour_idx, labels, min_bout_len, n_examples):
    Bout = namedtuple("Bout", ["start", "end"])
    labels = labels.astype("int")

    i, locs = -1, []
    while i < len(labels) - 1:
        i += 1
        
        if labels[i] != behaviour_idx: 
            i += 1
            continue
        
        j = i + 1
        while j < len(labels) - 1 and labels[i] == labels[j]: j += 1
        if j - i >= min_bout_len: locs.append(Bout(i, j-1))
        i = j
    
    locs.sort(key=lambda x: x.start - x.end)
    return locs[:n_examples]

def frames_for_bouts(video, locs):
    if not locs: return []
    locs.sort(key=lambda x: x.start)

    video = cv2.VideoCapture(video)
    fps = int(video.get(cv2.CAP_PROP_FPS))

    video = trim_video_data(video, fps, isvideo=True)
    
    idx, frames = 0, []
    
    success, image = video.read()  
    while success and locs:
        if idx == locs[0].start:
            while idx <= locs[0].end:
                frames.append(image)
                success, image = video.read()
                idx += 1
            for _ in range(fps): frames.append(np.zeros_like(image))
            locs.pop(0)
        else:
            success, image = video.read()
            idx += 1
    
    return frames
                

def behaviour_clips(behaviour_idx, videos, min_bout_len, fps, n_examples, clf, stride_window, bodyparts, conf_threshold, filter_thresh):
    min_bout_len = fps * min_bout_len // 1000
    clip_frames = []
    
    for video_file, video_name in videos:
        labels = video_frame_predictions(video_file, clf, stride_window, bodyparts, fps, conf_threshold, filter_thresh)
        locs = bouts_from_video(behaviour_idx, labels, min_bout_len, n_examples)    
        clip_frames.extend(frames_for_bouts(video_name, locs))
    
    return clip_frames
