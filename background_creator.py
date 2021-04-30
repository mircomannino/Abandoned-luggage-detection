'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''

import cv2
import numpy as np
from utils import *

class BackgroundCreator:
    '''
    Class used to manage the background model, with background subtraction.
    Attributes:
        foreground:     Estimated foreground model
        backSub:        Background subtractor model used to estimate the background
        learning_rate:  Learning rate used from the background model

    '''
    def __init__(self, method, learning_rate=-1):
        '''
        Constructor
        Args:
            method:         Method to use to estimate the background:
                            [MOG2, KNN]
            learning_rate:  Learning rate used from the background model
        '''
        # Initialize background subtractor
        self.foreground = None
        if method == 'MOG2':
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        elif method == 'KNN':
            self.backSub = cv2.createBackgroundSubtractorKNN()
        else:
            raise ValueError('Wrong Background subtraction method')

        # Assign learning_rate attribute
        self.learning_rate = learning_rate


    def update_background(self, frame):
        '''
        Method used to update the background model
        Args:
            frame:      The frame captured by the video frame sequence.
            learning_rate:  Learning rate of the background subtraction method
        '''
        self.foreground = self.backSub.apply(frame, self.foreground, self.learning_rate)

    def get_silhouette(self, mask):
        '''
        Method used to extract a silhouette of the stationary object according
        with the object in the given mask
        Args:
            mask:   Numpy tensor containing a mask of a particular category
                    of object (Ex: people)
        Returns:
            stationary_silhouette:  Numpy tensor with the silhouette of the
                                    stationary object in the given mask
        '''
        # People masks
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        boolean_mask = get_bool_mask(mask, threshold=0)

        # Foreground with invertend pixel values
        inverted_foreground = ~self.foreground
        _, inverted_foreground = cv2.threshold(inverted_foreground, 0, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), dtype='uint8')
        inverted_foreground =  cv2.morphologyEx(inverted_foreground, cv2.MORPH_ERODE, kernel)
        boolean_inverted_foreground = get_bool_mask(inverted_foreground, threshold=0)

        # And operation
        stationary_silhouette = np.logical_and(boolean_mask, boolean_inverted_foreground)
        stationary_silhouette = get_numerical_mask(stationary_silhouette, 255)

        # Apply morphological transformations
        stationary_silhouette = cv2.morphologyEx(stationary_silhouette, cv2.MORPH_DILATE, kernel)

        return  stationary_silhouette
