'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''

import numpy as np
import cv2
import argparse

def create_parser():
    '''
    Function used to parse the arguments of the command line for run script
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_file', '-i', type=str, default='./data/video1.mp4',
                        help='Path to the input video file, (default: ./data/video1.mp4)')
    parser.add_argument('--output_dir', '-o', type=str, default='./output/',
                        help='Path to the output folder (default: ./output/)')
    return parser.parse_args()

def get_box_from_indexes(pixels_indexes):
    '''
    Function to get a box from a set of indexes. The box is represented
    by 2 points (4 coordinates).
    Args:
        pixel_indexes:  Numpy tensor of shape (n_pixels, 2).
    Returns:
        box:            List of the coordinates of the points that represnt the box
    '''
    max_x, min_x = np.max(pixels_indexes[:, 1]), np.min(pixels_indexes[:, 1])
    max_y, min_y = np.max(pixels_indexes[:, 0]), np.min(pixels_indexes[:, 0])

    x1, y1 = min_x, min_y
    x2, y2 = max_x, max_y
    return [x1, y1, x2, y2]

def get_numerical_mask(bool_mask, value):
    '''
    Function used to transform a boolean mask into a mask with a given value in
    correspondence of the True value; The position with False value are filled with zeroes.
    Args:
        bool_mask:  Numpy tensor with boolean value
        value:      Value to put into the returned mask
    Returns:
        numerical_mask:     Numpy tensor (same dimension of bool_mask) with the
                            given value in correspondance of True value and zero otherwise
    '''
    numerical_mask = np.zeros(bool_mask.shape, dtype='uint8')
    numerical_mask[bool_mask == True] = value
    return numerical_mask

def get_bool_mask(numerical_mask, threshold):
    '''
    Conversion from numerical mask to bool mask. According with a threshold
    Args:
        numerical_mask:     Numpy tensor filled with number
        threshold:          Threshold to decide if in the returned mask there
                            will be False or True, according with the value in the
                            input mask
    Returns:
        bool_mask:          Numpy tensor with boolean values
    '''
    bool_mask = np.zeros(numerical_mask.shape, dtype='bool')
    bool_mask[numerical_mask > threshold] = True
    return bool_mask

def get_IoU(mask1, mask2):
    '''
    Function used to compute the Intersection Over Union score (IoU)
    Args:
        mask1:      Numpy tensor
        mask2:      Numpy tensor
    Returns:
        IoU:        IoU score between the two input masks
    '''
    IoU = 0.0
    correct_matrix = mask1[:, :] == mask2[:, :]
    n_correct_pixels = correct_matrix[correct_matrix==True].size # TP + TN
    # Compute the TN
    common_background = np.logical_and(mask1[:, :] == False, mask2[:, :] == False)
    TN = common_background[common_background==True].size
    # Compute IoU for the image
    TP = n_correct_pixels - TN
    n_total_pixels = mask1.shape[0] * mask1.shape[1]
    wrong_pixels = n_total_pixels - n_correct_pixels # FT + FN
    IoU += TP / (wrong_pixels + TP)
    return IoU

def get_reduced_silhouette(silhouette, box,  border_width_factor=2, border_height_factor=1.5):
    '''
    Method used to extract a sub-portion of the a silhouette image, according to a given box.
    Args:
        silhouette:             Numpy tensor with containing a silhouette
        box:                    4-points that represent the portion of the mask
                                that you want extract from the whole image
        border_width_factor:    Multiply factor used to scale the width of the given box
        border_height_factor:   Multiply factor used to scale the height of the given box
    Returns:
        reduced_silhouette:     Numpy tensor, representing the sup-portion
                                of the main_mask. It is in black&white.
                                The white pixels represent stationary pixels.
                                The black pixels represent non-stationary pixels.
    '''
    height = silhouette.shape[0]
    width = silhouette.shape[1]

    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    border_width = int((x2 - x1) * border_width_factor)
    border_height = int((y2 - y1) * border_height_factor)

    x1=np.amax([0,x1-border_width])
    y1=np.amax([0,y1-border_height])
    x2=np.amin([x2+border_width, width])
    y2=np.amin([y2+border_height, height])

    reduced_silhouette =  silhouette[y1:y2, x1:x2]
    return reduced_silhouette

def draw_single_bounding_box(frame, box, color):
    '''
    Function used to draw a box in a specific position in a frame.
    Args:
        frame:      Numpy tensor where the box will be drawn
        box:        List of coordinates that represent the box to be drawn
        color:      (BGR) format. Color of the box
    '''
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    thickness = 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, 'ABBANDONED', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)

def draw_all_bounding_boxes_from_clusters(frame, stationary_pixels_indexes, color):
    '''
    Function to draw a set of boxes in a frame, from a set of stationary pixels coordinates
    Args:
        frame:      Numpy tensor where the box will be drawn
        stationary_pixels_indexes:  List of sets of coordinates that represent stationary pixels
        color:      (BGR) format. Color of the boxes
    '''
    for stationary_pixels_index in stationary_pixels_indexes:
        box = get_box_from_indexes(stationary_pixels_index)
        draw_single_bounding_box(frame, box, color)

def draw_all_bounding_boxes_from_predictions(frame, predictions, indexes_stationary_object, category_id, color):
    '''
    Function to draw a set of boxes in a frame. This function select the box
    of the predictions to draw with the index contained in indexes_stationary_object.
    Args:
        frame:          Numpy tensor where the box will be drawn
        predictions:    Predictions given by detectron
        indexes_stationary_object:  Indexes of the predicted object that are highlighted
        category_id:    Category ID of the objects of interest
        color:          (BGR) format. Color of the boxes
    '''
    predicted_boxes = predictions['instances'].pred_boxes[predictions['instances'].pred_classes == category_id].tensor
    for index in indexes_stationary_object:
        draw_single_bounding_box(frame, predicted_boxes[index], color)
