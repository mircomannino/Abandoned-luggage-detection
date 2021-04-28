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
                            [MOG2, KNN, CNT]
            learning_rate:  Learning rate used from the background model
        '''
        # Initialize background subtractor
        self.foreground = None
        if method == 'MOG2':
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        elif method == 'KNN':
            self.backSub = cv2.createBackgroundSubtractorKNN()
        elif method == 'CNT':
            self.backSub = cv2.bgsegm.createBackgroundSubtractorCNT()
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
        # print('people: ', mask.dtype)
        # cv2_imshow(mask)
        boolean_mask = get_bool_mask(mask, threshold=0)

        # Foreground
        '''
        Abbiamo applicato erosione sulla foreground invertita, ma lo scopo è
        quello di andare a dilatare il foreground originale.
        Non applichiamo un'operazione morfologica direttamente all'immagine finale
        per non distorcere troppo le persone ferme.
        '''
        inverted_foreground = ~self.foreground
        _, inverted_foreground = cv2.threshold(inverted_foreground, 0, 255, cv2.THRESH_BINARY)
        # print('foreground: ', inverted_foreground.dtype)
        kernel = np.ones((5, 5), dtype='uint8')
        inverted_foreground =  cv2.morphologyEx(inverted_foreground, cv2.MORPH_ERODE, kernel)
        # print('cleaned foreground..............................................')
        # cv2_imshow(inverted_foreground)
        boolean_inverted_foreground = get_bool_mask(inverted_foreground, threshold=0)

        # And operation
        # print('tot TRUE people: ', np.sum(boolean_mask))
        # print('tot TRUE foreg: ', np.sum(boolean_inverted_foreground))
        stationary_silhouette = np.logical_and(boolean_mask, boolean_inverted_foreground)
        # print('tot TRUE final: ', np.sum(stationary_silhouette))
        # print('final type: ', stationary_silhouette.dtype)
        stationary_silhouette = get_numerical_mask(stationary_silhouette, 255)
        # print('final type: ', stationary_silhouette.dtype)

        # Apply morphological transformations
        '''
        TODO:
        Per evidenziare ancora meglio le persone dobbiamo applicare sul risultato
        finale (stationary_silhouette) un'operazione di dilatazione. Facendo così
        è come aver fatto una chiusura (o apertura) solo sulle persone ferme; ma
        abbiamo spezzato l'operazione in due fasi.
        '''
        stationary_silhouette = cv2.morphologyEx(stationary_silhouette, cv2.MORPH_DILATE, kernel)
        # cv2_imshow(stationary_silhouette)
        # kernel = np.ones((3, 3), dtype='uint8')
        # stationary_silhouette_cleaned = cv2.morphologyEx(stationary_silhouette, cv2.MORPH_OPEN, kernel)
        # cv2_imshow(stationary_silhouette_cleaned)

        return  stationary_silhouette
