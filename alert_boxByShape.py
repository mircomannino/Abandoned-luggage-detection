'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''

from stationary_objects_detector import *
from stationary_silhouette import *
from detectron2.engine import DefaultPredictor
import numpy as np

class Alert_BoxByShape:
    '''
    Class used to make the alert in case of abbandoned objects.
    It uses StationaryObjectsDetector objects in order to find stationary baggages
    It uses StationaryObjectsDetector objects in order to find stationary people
    It uses the shape of stationary object to draw the boxes of abbandoned object
    Attributes:
        alert_cfg:                      Configuration object of Alert Class
        stationary_threshold:           Threshold used to decide if a pixel is stationary (for objects)
        people_stationary_threshold:    Threshold used to decide if a pixel is stationary (for people)
        height:                         Height of the frame
        width:                          Width of the frame
        detectors:                      Dictionary of StationaryObjectsDetector objects. One detectors for each category to keep under control
        people_silhouette:              StationarySilhouette object used to manage stationary people
        predictor:                      Detectron2 predictor. Used to predict objects in each frame
    '''
    def __init__(self, height, width, fps, alert_cfg):
        '''
        Constructor
        Args:
            height:                     Height of the frame
            width:                      Width of the frame
            fps:                        FPS of the original video
            alert_cfg:                  Configuration object. Used to setup some parameters
        '''
        self.alert_cfg = alert_cfg
        self.stationary_threshold = np.min([(fps * self.alert_cfg.STATIONARY_SECONDS), 254])
        self.people_stationary_threshold = np.min([(fps * self.alert_cfg.PEOPLE_STATIONARY_SECONDS), 255])
        self.height = height
        self.width = width
        self.detectors = {}
        for category_id in self.alert_cfg.CATEGORIES_ID:
            self.detectors[category_id] = StationaryObjectsDetector(self.height, self.width, category_id=category_id, tot_frames_sliding_window=(self.alert_cfg.SLIDING_WINDOW_SECONDS*fps), color=self.alert_cfg.COLORS[category_id])
        self.people_silhouette = StationarySilhouette(self.height, self.width, category_id=self.alert_cfg.PEOPLE_ID, tot_frames_sliding_window=(self.alert_cfg.SLIDING_WINDOW_SECONDS*fps))
        self.predictor = DefaultPredictor(self.alert_cfg.detectron_cfg)


    def notify_alert(self, frame):
        '''
        Method used to check if objects are abbandoned and to highlight
        which are the abbandoned objects through a colored box arround the abbandoned objects.
        In this version the stationary object boxes are drawn using only the
        stationary pixels of the objects.
        Args:
            frame:  The frame captured by the video frame sequence.
        '''
        predictions = self.predictor(frame)

        self.people_silhouette.add_predictions(predictions, threshold_accumulation_mask=self.alert_cfg.THRESHOLD_ACCUMULATION_MASK, back_step=self.alert_cfg.BACK_STEP)

        for category_id in self.detectors:
            # Add the prediction to the detector
            self.detectors[category_id].add_predictions(predictions, threshold_accumulation_mask=self.alert_cfg.THRESHOLD_ACCUMULATION_MASK, back_step=self.alert_cfg.BACK_STEP)

            # Analyze the predictions
            stationary_pixels_indexes = self.detectors[category_id].find_clusters_pixels_indexes(stationary_threshold=self.stationary_threshold)

            if len(stationary_pixels_indexes) > 0:
                # Search stationary people in the whole mask
                if (self.people_silhouette.accumulation_mask.find_stationary_pixels(self.people_stationary_threshold)):
                    for stationary_pixels_index in stationary_pixels_indexes:
                        if (len(stationary_pixels_index) < (self.width*self.height)*self.alert_cfg.NOISE_SCALE_FACTOR_BAGGAGE_SILHOUETTE):
                            continue    # Cluster of pixels is to small, skip it
                        box = get_box_from_indexes(stationary_pixels_index)
                        people_window_black_white = self.people_silhouette.get_reduced_main_mask(box, self.people_stationary_threshold)
                        # compute the noise threshold
                        noise_threshold = (people_window_black_white.shape[0] * people_window_black_white.shape[1]) * self.alert_cfg.NOISE_SCALE_FACTOR_PEOPLE_SILHOUETTE_REDUCED
                        # Search if there are stationary pepople in the window
                        if (people_window_black_white > 0).sum() < noise_threshold:
                            print('ALARM: Abbandoned baggage')
                            draw_single_bounding_box(frame, box, self.detectors[category_id].color)
                        else:
                            print('Find person near a stationary object')
                else:
                    print('ALARM: Abbadoned baggage')
                    draw_all_bounding_boxes_from_clusters(frame, stationary_pixels_indexes, self.detectors[category_id].color)
