'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''

from sliding_window import SlidingWindow
from accumulation_mask import AccumulationMask
import numpy as np
import cv2

class StationarySilhouette:
    '''
    Class used to manage the construction of a silhouette for a particular
    category of objects. It combines slidingWindow and AccumulationMask
    Attributes:
        heigth:             Height of the images
        width:              Width of the images
        category_id:        Id of the category of interest (According to COCO dataset)
        sliding_window:     SlidingWindow object
        accumulation_mask:  AccumulationMask object
    '''
    def __init__(self, height, width, category_id, tot_frames_sliding_window):
        '''
        Constructor
        Args:
            heigth:             Height of the images
            width:              Width of the images
            category_id:        Id of the class of interest (According to COCO dataset)
            tot_frames_sliding_window:  Total number of frames in the SlidingWindow object
        '''
        self.height = height
        self.width = width
        self.category_id = category_id
        self.sliding_window = SlidingWindow(self.height, self.width, tot_frames=tot_frames_sliding_window)
        self.accumulation_mask = AccumulationMask(self.height, self.width)

    def add_predictions(self, predictions, threshold_accumulation_mask, back_step):
        '''
        Method used to update the internal attributes of the StationarySilhouette
        object according to a new prediction.
        Args:
            predictions:    Dictionary object with all the predictions provdided by Detectron
            threshold_accumulation_mask:    Threshold for the accumulationMask object
            back_step:      Back step for the accumulationMask object
        '''
        all_masks = np.sum(predictions['instances'].pred_masks[predictions['instances'].pred_classes == self.category_id].cpu().numpy(), axis=0)
        self.sliding_window.add_frame(all_masks)
        self.accumulation_mask.add_mask(mask=self.sliding_window.get_mean_frame(), threshold=threshold_accumulation_mask, back_step=back_step)

    def get_reduced_main_mask(self, box, stationary_threshold, border_width_factor=2, border_height_factor=1.5):
        '''
        Method used to extract a sub-portion of the main mask, according to a given box.
        The output mask is black&white, according to a given stationary threshold.
        The white pixels represent stationary pixels.
        The black pixels represent non-stationary pixels.
        Args:
            box:                    [x1, y1, x2, y2] values that represent the portion of the mask
                                    that you want extract from the whole mask
            stationary_threshold:   Threshold on pixels values, used to evaluate
                                    the stationary of each pixel
            border_width_factor:    Multiply factor used to scale the width of the given box
            border_height_factor:   Multiply factor used to scale the height of the given box
        Returns:
            black_white_mask:       Numpy tensor, representing the sup-portion
                                    of the main_mask. It is in black&white.
                                    The white pixels represent stationary pixels.
                                    The black pixels represent non-stationary pixels.
        '''
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        border_width = int((x2 - x1) * border_width_factor)
        border_height = int((y2 - y1) * border_height_factor)

        x1=np.amax([0,x1-border_width])
        y1=np.amax([0,y1-border_height])
        x2=np.amin([x2+border_width, self.width])
        y2=np.amin([y2+border_height,self.height])

        ret, black_white_mask = cv2.threshold(self.accumulation_mask.main_mask, stationary_threshold, 255, cv2.THRESH_BINARY)
        black_white_mask = black_white_mask[y1:y2, x1:x2]

        return black_white_mask
