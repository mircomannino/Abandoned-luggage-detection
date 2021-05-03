'''
Authors:    Federico Giusti, Mirco Mannino, Silvia Palmucci
Project:    Abandoned objects detection
'''

from stationary_silhouette import StationarySilhouette
import numpy as np
import cv2
from utils import *
from sklearn.cluster import MiniBatchKMeans

class StationaryObjectsDetector(StationarySilhouette):
    '''
    Class used to manage the stationary objects for a particular category.
    It is derived from StationarySilhouette.
    Attributes (from StationarySilhouette):
        height:             Height of the images
        width:              Width of the images
        category_id:        Id of the category of interest (According to COCO dataset)
        sliding_window:     SlidingWindow object
        accumulation_mask:  AccumulationMask object
    Attributes:
        color:              Color (BGR format) with which the abandoned objects
                            are highlighted
    '''
    def __init__(self, height, width, category_id, tot_frames_sliding_window, color):
        '''
        Constructor
        Args:
            heigth:             Height of the images
            width:              Width of the images
            category_id:        Id of the class of interest (According to COCO dataset)
            tot_frames_sliding_window:  Total number of frames in the SlidingWindow object
            color:              BGR color
        '''
        super().__init__(height, width, category_id, tot_frames_sliding_window)
        self.color = color


    def find_stationary_predictions_indexes(self, predictions, stationary_threshold):
        '''
        Method used to select the indexes (in the predictions) that are
        stationary. These indexes will be used to check if the correspondance
        object is abandoned.
        Args:
            predictions:    Dictionary returned by detectron
            stationary_threshold:   Threshold which determines if a pixel is
                                    stationary. If the pixel value is greater
                                    than the threshold, the pixel is stationary
        Returns:
            Numpy tensor with all the indexes of the stationary objects
            (According with index in predictions)
        '''
        predicted_masks = predictions['instances'].pred_masks[predictions['instances'].pred_classes == self.category_id].cpu().numpy()

        if len(predicted_masks) == 0:
            return np.array([])

        if not self.accumulation_mask.find_stationary_pixels(stationary_threshold):
            return np.array([])

        # Find number of stationary objects
        ret, black_white_empirical_mask = cv2.threshold(self.accumulation_mask.main_mask, stationary_threshold, 255, cv2.THRESH_BINARY)
        black_white_empirical_mask = black_white_empirical_mask.astype('uint8') # Aggiunta da Mirco per fare la conversione del tipo della maschera
        bool_empirical_mask = get_bool_mask(self.accumulation_mask.main_mask, threshold=stationary_threshold)
        n_empirical_instances = self.__get_number_of_stationary_objects(black_white_empirical_mask)

        # Return empty array if there are not stationary objects
        if n_empirical_instances == 0:
            return np.array([])

        # Find clusters
        bool_empirical_mask = get_bool_mask(self.accumulation_mask.main_mask, threshold=stationary_threshold)
        clusterized_masks, _ = self.__find_clusters(bool_empirical_mask=bool_empirical_mask, n_clusters=n_empirical_instances)

        # Find IoU scores
        indexes_stationary_object = np.array([])
        for single_object in clusterized_masks:
            bool_single_object = get_bool_mask(single_object, threshold=254)
            IoUs = np.array([])
            for predicted_mask in predicted_masks:
                # Try to apply a threshold for the IoU value
                cur_IoU = get_IoU(predicted_mask, bool_single_object)
                if (cur_IoU > 0.2):
                    IoUs = np.append(IoUs, cur_IoU)
                else:
                    IoUs = np.append(IoUs, -1)
            if (IoUs == -1).sum() < IoUs.size: # Se ho almeno un valore diverso da -1 aggiungo l'indice al risultato
                indexes_stationary_object = np.append(indexes_stationary_object, np.argmax(IoUs))
        return np.unique(np.array(indexes_stationary_object, dtype='int'))

    def find_clusters_pixels_indexes(self, stationary_threshold):
        '''
        Method used to find all the clusters of pixels that are stationary, they
        represent the abandoned objects.
        Args:
            stationary_threshold:   Threshold to evaluate if a pixel is stationary
        Returns:
            clusters_pixels_indexes:    List of clusters. Each element
                                        of the list represent a stationary object
                                        and it is represented by a set of coordinates
                                        that belong to the object.
                                        The shape of each element is: (n_pixels, 2)
        '''

        if not self.accumulation_mask.find_stationary_pixels(stationary_threshold):
            return np.array([])

        # Find number of stationary objects
        ret, black_white_empirical_mask = cv2.threshold(self.accumulation_mask.main_mask, stationary_threshold, 255, cv2.THRESH_BINARY)
        black_white_empirical_mask = black_white_empirical_mask.astype('uint8')
        bool_empirical_mask = get_bool_mask(self.accumulation_mask.main_mask, threshold=stationary_threshold)
        n_empirical_instances = self.__get_number_of_stationary_objects(black_white_empirical_mask)

        if n_empirical_instances == 0:
            return np.array([])

        # Find clusters
        bool_empirical_mask = get_bool_mask(self.accumulation_mask.main_mask, threshold=stationary_threshold)
        clusterized_masks, clusters_pixels_indexes = self.__find_clusters(bool_empirical_mask=bool_empirical_mask, n_clusters=n_empirical_instances)

        return clusters_pixels_indexes

    def __find_clusters(self, bool_empirical_mask, n_clusters):
        '''
        Internal method used to find clusters of stationary pixels in the
        whole mask. This method uses KMeans clustering algorithm
        Args:
            bool_empirical_mask:    Numpy tensor containing True/False values.
                                    True if a pixel value is greater than the stationary threshold
                                    False if a pixel value is less than the stationary threshold
        Returns:
            clusterized_masks:      Numpy tensor containing a mask for each
                                    cluster of stationary pixel detected.
                                    shape: (n_cluster, heigth, width)
            stationary_pixels_indexes:  List of clusters. Each element
                                        of the list represent a stationary object
                                        and it is represented by a set of coordinates
                                        that belong to the object.
                                        The shape of each element is: (n_pixels, 2)
        '''

        stationary_pixels_indexes = []

        stationary_pixels = np.argwhere(bool_empirical_mask == True)
        clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=0,  batch_size=1000).fit(stationary_pixels)
        labels = clustering.labels_
        clusterized_masks = np.array([])
        for k in range(n_clusters):
            single_cluster = np.zeros((self.height, self.width))
            cluster_index = stationary_pixels[np.where(labels==k)] # Prendo tutti gli indici del cluster corrente

            stationary_pixels_indexes.append(cluster_index)

            single_cluster[cluster_index[:,0], cluster_index[:, 1]] = 255
            if clusterized_masks.size == 0:
                clusterized_masks = single_cluster.reshape(1, self.height, self.width)
            else:
                clusterized_masks = np.append(clusterized_masks, single_cluster).reshape(k+1, self.height, self.width)
        return clusterized_masks, stationary_pixels_indexes


    def __get_number_of_stationary_objects(self, mask):
        '''
        Internal method used to count the number of clusters, used by the
        Kmeans clustering algorithm. This method uses the finContours function
        provided by OpenCV to the cleaned mask. The mask is cleaned with
        some morphological operations.
        Args:
            mask:   Numpy tensor with all the stationary_objects.
        Returns:
            n_stationary_objects:   Number of objects find into the mask
        '''
        # Apply morphological transformations
        kernel = np.ones((5, 5), dtype='uint8')
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Find contours
        contours = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Filter objects that are too small
        n_stationary_objects = 0
        for contour in contours[0]:
            if contour.shape[0] > 50:
                n_stationary_objects += 1
        return n_stationary_objects
