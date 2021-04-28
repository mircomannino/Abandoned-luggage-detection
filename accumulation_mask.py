'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''
import numpy as np

class AccumulationMask:
    '''
    Class used to manage the accumulation of multiple masks into a single one
    Attributes:
        height:     Height of the masks
        width:      Width of the masks
        main_maisk: Numpy array that represent the accumulation mask
    '''
    def __init__(self, height, width):
        '''
        Constructor
        Args:
            height:     Height of the masks
            width:      Width of the masks
        '''
        self.height = height
        self.width = width
        self.main_mask = np.zeros((height, width), dtype='float64')

    def add_mask(self, mask, threshold=0.5, min_val=0, max_val=255, back_step=2):
        '''
        Method used to add values to the pixel of stationary objects.
        All the operations are made in-place
        Args:
            mask:       Mask of values which identifies the stationary objects (Tensor).
            threshold:  Threshold for the values of the mask. To avoid noise.
            min_val:    Minimum value allowed in the main_img tensor (real application: 0).
            max_val:    Maximum vaule allowed in the main_img tensor (real application: 255).
            back_step:  Value that is subtracted in case in which the value of a
                        pixel is less than the threshold
        '''
        # Apply threshold to the mask
        self.main_mask[mask >= threshold] += 1
        self.main_mask[mask < threshold] += (-1) * back_step

        # Deal with overflow and underflow (min=0, max=255)
        self.main_mask[self.main_mask < min_val] = min_val
        self.main_mask[self.main_mask > max_val] = max_val

    def find_stationary_pixels(self, stationary_threshold):
        '''
        Find if there are pixels that are greater than the threshold
        Args:
            stationary_threshold:   Threshold to be exceeded in order to be stationary
        Returns:
            [True/False]:   True if there are pixels with value greater
                            than the stationary_threshold. False otherwise.
        '''
        return len(self.main_mask[self.main_mask > stationary_threshold]) > 0
