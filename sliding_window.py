'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''

import numpy as np

class SlidingWindow:
    '''
    Class to manage the sliding window
    Attributes:
        height:     Height of the frames
        width:      Width of the frames
        tot_frames: Number of max number of frames in the window
        mean_frame: Single frame obtained with the mean of all the frames
    '''
    def __init__(self, height, width, tot_frames=60):
        '''
        Constructor
        Args:
            height:     Height of the frames
            width:      Width of the frames
            tot_frames: Number of max number of frames in the window
        '''
        self.height = height
        self.width = width
        self.tot_frames = tot_frames
        self.frames = np.array([])

    def get_mean_frame(self):
        '''
        Method used to get the mean frame of the sliding window
        '''
        return np.mean(self.frames, axis=0)

    def add_frame(self, frame_to_add):
        '''
        Method used to add a new frame in the sliding window
        Args:
            frame_to_add:   New frame to add in the sliding window
        '''
        actual_n_frames = self.frames.shape[0]
        if (actual_n_frames + 1) <= self.tot_frames:
            actual_n_frames += 1
            self.frames = np.append(self.frames, frame_to_add).reshape(actual_n_frames, self.height, self.width)
        else:
            # Manage the overflow of frames
            self.frames = np.append(self.frames[1:], frame_to_add).reshape(actual_n_frames, self.height, self.width)
