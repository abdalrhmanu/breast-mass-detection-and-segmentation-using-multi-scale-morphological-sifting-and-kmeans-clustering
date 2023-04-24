import sys
sys.path.append('..\\helpers')

import cv2
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

from PIL import Image, ImageDraw
from functools import reduce
from pywt import dwt2
from scipy import ndimage
from loguru import logger
from tqdm import tqdm, tqdm_notebook

# From helpers directory
from preprocessing import Preprocessor
import display


class MorphologicalSifter:
    def __init__(self):
        self.preprocessor = Preprocessor()     
        self.image_dir = os.path.dirname("../dataset/processed/images/")   
        self.overlay_dir = os.path.dirname("../dataset/processed/overlay/") 
        self.area_min = 15
        self.area_max = 3689
        self.mass_size_range = [self.area_min, self.area_max] # square mm
        self.pixel_size = 0.07 # spatial resolution of the INbreast mammograms, 0.07mm
        self.resize_ratio = 1 / self.preprocessor.scale_factor
        self.n_scale = 2
        self.n_lse_elements = 18
        self.lse = None
        self.angle_range = None
        self.mass_diameter_range_pixel = \
            [math.floor((self.mass_size_range[0]/math.pi)**0.5*2/(self.pixel_size/self.resize_ratio)),
                                    math.ceil((self.mass_size_range[1]/math.pi)**0.5*2/(self.pixel_size/self.resize_ratio))]  # diameter range in pixels
        
    def _subsample_image(self, image):
        # Image subsampling using 2 level db2 wavelet
        image = image[:,:,0]
        breast_mask = (image > 0)
        (cA, _) = dwt2(image, 'db2')
        (image, _) = dwt2(cA, 'db2')

        (cA, _) = dwt2(breast_mask, 'db2')
        (breast_mask, _) = dwt2(cA, 'db2')
        breast_mask = (breast_mask >= 1)

        return image, breast_mask
    
    def _linear_structuring_elements(self, Num_scale, D):
        """
        This function generates the length of the linear structuring elements (LSE) 
        used in morphological filter elements on different scales. Either linear or 
        logarithmic scale interval is used.

        INPUT:
        Num_scale : int : The number of scales used
        D : list : The diameter range of breast masses

        OUTPUT:
        len_bank : numpy.ndarray : The magnitudes of the LSEs
        """
        scale_interval = (D[1] / D[0]) ** (1 / Num_scale)
        len_bank = np.zeros(Num_scale + 1, dtype=int)
        for l in range(Num_scale + 1):
            len_bank[l] = int(D[0] * (scale_interval ** (l - 1)))
        len_bank[Num_scale] = D[1]

        return len_bank
    
    def _normalize(self, image, mask, astype):
        image = (image-image.min())/(image.max()-image.min())

        if astype == 8:
            image *= 255
            image = image.astype(np.uint8)
            image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        elif astype == 16:
            image *= 65535
            image = image.astype(np.uint16)
            image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint16))
        else:
            raise ValueError('Not supported type')
        
        return image
    
    def _multiscale_morphological_sifter(self, M1, M2, orientation, image, breast_mask):
        # Create a copy of the input image
        newimage = np.copy(image)
        
        # Get the dimensions of the input image
        m, n = newimage.shape
        
        # Pad with highest pixel value
        temp = np.full((m+4*M1, n+4*M1), 65535, dtype=np.uint16)
        temp[2*M1:(2*M1+m), 2*M1:(2*M1+n)] = newimage
        
        # Apply multi-scale morphological sifting
        enhanced_image = np.zeros_like(temp)
        for k in range(len(orientation)):
            B1 = cv2.getStructuringElement(cv2.MORPH_RECT, (M1, 1), (-1, -1))
            B2 = cv2.getStructuringElement(cv2.MORPH_RECT, (M2, 1), (-1, -1))
            bg1 = cv2.morphologyEx(temp, cv2.MORPH_OPEN, B1)
            r1 = cv2.subtract(temp, bg1)
            r2 = cv2.morphologyEx(r1, cv2.MORPH_OPEN, B2)
            enhanced_image = enhanced_image + r2.astype(np.float64)

        enhanced_image = enhanced_image[2*M1:2*M1+m, 2*M1:2*M1+n] # Reset the image into the original size

        enhanced_image = (enhanced_image-enhanced_image.min())/(enhanced_image.max()-enhanced_image.min())*255
        enhanced_image = enhanced_image.astype(np.uint8)
        enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=breast_mask.astype(np.uint8))
        
        return enhanced_image



    def fit(self, image_input_name, plot = False, test_on_overlay=False):

        if(test_on_overlay):
            input_image = cv2.imread(os.path.join(self.overlay_dir, image_input_name))
        else:
            input_image = cv2.imread(os.path.join(self.image_dir, image_input_name))

        if input_image is None:
            raise ValueError("Image doesn't exist.")

        self.lse = self._linear_structuring_elements(self.n_scale, self.mass_diameter_range_pixel)
        self.angle_range = list(range(0, 190, self.n_lse_elements))

        image_subsampled, breast_mask_subsampled = self._subsample_image(input_image)
        image_normalized = self._normalize(image=image_subsampled, mask=breast_mask_subsampled, astype=8)

        enhanced_image = []

        for i in range(1, self.n_scale+1):
            enhanced_image.append(
                self._multiscale_morphological_sifter(
                    self.lse[i], 
                    self.lse[i-1], 
                    self.angle_range, 
                    image_normalized, 
                    breast_mask_subsampled))
            
        enhanced_image = np.sum(enhanced_image, axis=0)

        # applying a grayscale normalization (to 16-bit) on the summation of all the result images generated
        # Perform normalization using clip and interp
        enhanced_image = enhanced_image.astype('float')

        normalized_image = np.interp(
            np.clip(enhanced_image, enhanced_image.min(), enhanced_image.max()), 
            [enhanced_image.min(), enhanced_image.max()], [0, 65535]).astype('uint16')



        if plot:
            imgs = {
                "Original Image": input_image,
                "Enhanced Image": normalized_image
            }

            display.plot_figures(imgs, 1,2) 
        
        return normalized_image



