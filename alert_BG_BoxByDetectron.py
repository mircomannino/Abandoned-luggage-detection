'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''

from stationary_objects_detector import *
from background_creator import *
from detectron2.engine import DefaultPredictor
import torch
import numpy as np

class Alert_BG_BoxByDetectron:
    '''
    Class used to make the alert in case of abbandoned objects.
    It uses StationaryObjectsDetector objects in order to find stationary baggages
    It uses a background model in order to find stationary people.
    It uses boxes of detectron prediction to draw the box of abbandoned object
    Attributes:
        alert_cfg:                      Configuration object of Alert Class
        stationary_threshold:           Threshold used to decide if a pixel is stationary (for objects)
        height:                         Height of the frame
        width:                          Width of the frame
        detectors:                      Dictionary of StationaryObjectsDetector objects. One detectors for each category to keep under control
        background:                     BackgorundCreator object, used to manage the background model
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
        self.height = height
        self.width = width
        self.detectors = {}
        for category_id in self.alert_cfg.CATEGORIES_ID:
            self.detectors[category_id] = StationaryObjectsDetector(self.height, self.width, category_id=category_id, tot_frames_sliding_window=(self.alert_cfg.SLIDING_WINDOW_SECONDS*fps), color=self.alert_cfg.COLORS[category_id])
        self.background = BackgroundCreator(self.alert_cfg.BACKGROUND_METHOD, self.alert_cfg.BACKGROUND_LEARNING_RATE)
        self.predictor = DefaultPredictor(self.alert_cfg.detectron_cfg)


    def notify_alert(self, frame):
        '''
        Method used to check if objects are abbandoned and to highlight
        which are the abbandoned objects through a colored box arround the abbandoned objects.
        In this version the stationary object boxes are drawn using the boxes
        of the predicted objects.
        Args:
            frame:  The frame captured by the video frame sequence.
        '''
        predictions = self.predictor(frame)
        self.background.update_background(frame)

        # Find people silhouette
        people_masks = np.sum(predictions['instances'].pred_masks[predictions['instances'].pred_classes == self.alert_cfg.PEOPLE_ID].cpu().to(torch.float64).numpy(), axis=0)
        people_silhouette = self.background.get_silhouette(people_masks)

        for category_id in self.detectors:
            # Add the prediction to the detector
            self.detectors[category_id].add_predictions(predictions, threshold_accumulation_mask=self.alert_cfg.THRESHOLD_ACCUMULATION_MASK, back_step=self.alert_cfg.BACK_STEP)

            # Analyze the predictions
            predicted_stationary_indexes = self.detectors[category_id].find_stationary_predictions_indexes(predictions, stationary_threshold=self.stationary_threshold)

            if (len(predicted_stationary_indexes) > 0):
                noise_threshold_image = frame.shape[0] * frame.shape[1] * self.alert_cfg.NOISE_SCALE_FACTOR_PEOPLE_SILHOUETTE
                if (people_silhouette > 0).sum() > noise_threshold_image:   # Check if there are stationary people

                    predicted_boxes = predictions['instances'].pred_boxes[predictions['instances'].pred_classes == category_id].tensor
                    predicted_boxes = predicted_boxes[predicted_stationary_indexes]

                    for index, box in enumerate(predicted_boxes):

                        people_silhouette_window = get_reduced_silhouette(people_silhouette, box, border_width_factor=2, border_height_factor=1.5)

                        # compute the noise threshold
                        noise_threshold_window = (people_silhouette_window.shape[0] * people_silhouette_window.shape[1]) * self.alert_cfg.NOISE_SCALE_FACTOR_PEOPLE_SILHOUETTE_REDUCED
                        # Search if there are stationary pepople in the window
                        if (people_silhouette_window > 0).sum() < noise_threshold_window:  # 50 pixel di rumore sono ammessi
                            print('ALARM: Abbandoned baggage')
                            draw_single_bounding_box(frame, box, self.detectors[category_id].color)
                        else:
                            print('Find person near a stationary object')
                else:
                    print('ALARM: Abbadoned baggage')
                    draw_all_bounding_boxes_from_predictions(frame, predictions, predicted_stationary_indexes, category_id, self.detectors[category_id].color)
