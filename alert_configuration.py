'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import get_cfg
import torch

class AlertConfiguration:
    '''
    class used to manage all the parameters of the Alert class
    Attributes:
        STATIONARY_SECONDS:
        Seconds after which one object is considered abbandoned

        PEOPLE_STATIONARY_SECONDS:
        Seconds after which a person is considered stationary

        SLIDING_WINDOW_SECONDS:
        Width of the sliding window in seconds. This number will be multiplied
        with the FPS, in order to obtain the number of frame in the sliding window

        THRESHOLD_ACCUMULATION_MASK:
        Used in AccumulationMask. This is tha value to establish if a pixel belong
        or not to a certain object mask.

        BACK_STEP:
        Used in AccumulationMask. This is the value by which
        a pixel is decremented in the case in which the value of this pixel
        is less than THRESHOLD_ACCUMULATION_MASK.

        SKIPPED_FRAMES:
        Number of frames of the input video that have to be skipped during 
        video processing.

        PEOPLE_ID:
        Id of the people category. According with COCO dataset IDs.

        CATEGORIES_ID:
        List of the IDs of the category to keep under control. According with
        COCO dataset IDs.

        COLORS:
        Dictionary with the color of each category.

        NOISE_SCALE_FACTOR_BAGGAGE_SILHOUETTE:
        Scale factor multiplied with the area of the frame.
        This multiplication is used to setup the noise threshold, in order
        to distinguish a real stationary object or only noise

        NOISE_SCALE_FACTOR_PEOPLE_SILHOUETTE:
        Scale factor multiplied with the area of the frame.
        This multiplication is used to setup the noise threshold, in order
        to distinguish a real stationary person or only noise

        NOISE_SCALE_FACTOR_PEOPLE_SILHOUETTE_REDUCED:
        Scale factor multiplied with the area of a window in which there is a person.
        This multiplication is used to setup the noise threshold, in order
        to distinguish a window where there is a stationary person or only noise

        BACKGROUND_METHOD:
        Method of background subtraction. Used only in Alert_BG_BoxByDetectron.
        Possible choices: [MOG2, KNN]

        BACKGROUND_LEARNING_RATE:
        Learning rate of the background model

        detectron_cfg:
        It is the configuration for detectron2. It takes pretrained weights
        and set the threshold for the predictions.
    '''
    def __init__(self):
        self.STATIONARY_SECONDS = 30
        self.PEOPLE_STATIONARY_SECONDS = 10
        self.SLIDING_WINDOW_SECONDS = 10
        self.BACK_STEP = 1
        self.SKIPPED_FRAMES = 2
        self.THRESHOLD_ACCUMULATION_MASK = 0.5
        self.PEOPLE_ID = 0
        self.CATEGORIES_ID = [24, 26, 28]   # backpack, handbag, suitcase
        self.COLORS = {24: (65, 14, 240), 26: (33, 196, 65), 28: (236, 67, 239)}
        self.NOISE_SCALE_FACTOR_BAGGAGE_SILHOUETTE = 5e-4
        self.NOISE_SCALE_FACTOR_PEOPLE_SILHOUETTE = 1e-3
        self.NOISE_SCALE_FACTOR_PEOPLE_SILHOUETTE_REDUCED = 7e-4

        self.BACKGROUND_METHOD = 'MOG2'
        self.BACKGROUND_LEARNING_RATE = 0.0007

        # Make detectron configuration
        self.detectron_cfg = get_cfg()
        self.detectron_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for the model
        self.detectron_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        # Select the device
        if not torch.cuda.is_available():
            self.detectron_cfg.MODEL.DEVICE = 'cpu'
            print('device: cpu')
        else:
            print('device: cuda')

    def __str__(self):
        return_str = '\n*************** Alert configuration **************\n'
        for attributes, val in self.__dict__.items():
            if(attributes != "detectron_cfg"):
                return_str += '\t' + str(attributes) + ': ' + str(val) + '\n'
        return_str += '**************************************************'
        return return_str
