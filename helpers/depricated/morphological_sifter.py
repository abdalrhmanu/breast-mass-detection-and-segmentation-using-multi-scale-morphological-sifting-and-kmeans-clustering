import sys
sys.path.append('..\\helpers')

import cv2
import os
import numpy as np
import math

from functools import reduce
from pywt import dwt2
from tqdm import tqdm, tqdm_notebook

# From helpers directory
import display


class MorphologicalSifter:
    def __init__(self):
        self.image_dir = os.path.dirname("../dataset/processed/images/")   
        self.overlay_dir = os.path.dirname("../dataset/processed/overlay/") 
        self.area_min = 15
        self.area_max = 3689
        self.mass_size_range = [self.area_min, self.area_max] # square mm
        self.pixel_size = 0.07 # spatial resolution of the INbreast mammograms, 0.07mm
        self.resize_ratio = 1 / 10
        self.n_scale = 4
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
            # len_bank[l] = round(D[0] * (scale_interval ** (l - 1)))
            len_bank[l] = round(D[0] * (scale_interval ** (l)))
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
    
    def _multiscale_morphological_sifter(self, M1, M2, orientation, image, breast_mask, L_OR_R, padding_option):
        # Create a copy of the input image        
        newimage = image.copy()
        m, n = newimage.shape

        ## Border effect control: border padding
        # Option 1: pad with highest pixel value
        temp = np.full((m + 4 * M1, n + 4 * M1), 65535, dtype=np.uint16)
        temp[2 * M1:2 * M1 + m, 2 * M1:2 * M1 + n] = newimage  # Add white margins to each side of the image to prevent edge effect of morphological process


        # Option 2: replicate the pixels on the border
        if padding_option == 1:
            if L_OR_R == 1:  # left breast
                edge = newimage[:, :min(n, 2 * M1)]
                temp[2 * M1:2 * M1 + m, 2 * M1 - edge.shape[1]:2 * M1] = np.fliplr(edge)
            else:  # right breast
                edge = newimage[:, max(1, n - 2 * M1 + 1) - 1:n]
                temp[2 * M1:2 * M1 + m, n + 2 * M1:n + 2 * M1 + edge.shape[1]] = np.fliplr(edge)
        
        # Apply multi-scale morphological sifting
        enhanced_image = np.zeros(temp.shape, dtype=np.float32)
        for k in range(len(orientation)):
            B1 = cv2.getStructuringElement(cv2.MORPH_RECT, (M1, 1), anchor=(0, 0))
            # B1 = cv2.rotate(B1, orientation[k])
            B2 = cv2.getStructuringElement(cv2.MORPH_RECT, (M2, 1), anchor=(0, 0))
            # B2 = cv2.rotate(B2, orientation[k])
            bg1 = cv2.morphologyEx(temp, cv2.MORPH_OPEN, B1)
            r1 = cv2.subtract(temp, bg1)
            r2 = cv2.morphologyEx(r1, cv2.MORPH_OPEN, B2)
            enhanced_image = enhanced_image + r2.astype(np.float64)

        enhanced_image = enhanced_image[2 * M1:2 * M1 + m, 2 * M1:2 * M1 + n]  # Reset the image into the original size

        # Normalization masking
        enhanced_image = self._normalize(enhanced_image, breast_mask, 8)

        # Crop the white margins from the bottom of the enhanced image
        enhanced_image = enhanced_image[:m, :n]
        
        return enhanced_image



    def fit(self, image_input_name, plot = False, test_on_overlay=False):

        if(test_on_overlay):
            input_image = cv2.imread(os.path.join(self.overlay_dir, image_input_name))
        else:
            input_image = cv2.imread(os.path.join(self.image_dir, image_input_name))

        if input_image is None:
            raise ValueError("Image doesn't exist.")
        
        self.lse = self._linear_structuring_elements(self.n_scale, self.mass_diameter_range_pixel)
        self.angle_range = list(range(0, 180, 10))

        image_subsampled, breast_mask_subsampled = self._subsample_image(input_image)
        image_normalized = self._normalize(image_subsampled, breast_mask_subsampled,8)

        enhanced_images = []  
        
        L_OR_R = 0 if '_R_' in image_input_name else 1 # check if it is a left or right breast
        CC_OR_ML = 1 if '_CC_' in image_input_name else 0

        for j in range(1, self.n_scale+1):
            # Boundary padding
            padding_mode = 1
            if j==1 or CC_OR_ML==1:
                # if it is a small scale or it is a MLO view
                padding_mode = 0  # highest value padding

            enhanced_images.append(
                self._multiscale_morphological_sifter(
                    self.lse[j], 
                    self.lse[j-1], 
                    self.angle_range, 
                    image_normalized, 
                    breast_mask_subsampled,
                    L_OR_R,
                    padding_mode))
            
        summed_image = np.sum(enhanced_images, axis=0)

        # applying a grayscale normalization (to 16-bit) on the summation of all the result images generated
        # Perform normalization using clip and interp
        summed_image = summed_image.astype('float')

        normalized_image = np.interp(
            np.clip(summed_image, summed_image.min(), summed_image.max()), 
            [summed_image.min(), summed_image.max()], [0, 65535]).astype('uint16')
        

        if plot:
            imgs = {
                "Original Image": input_image,
                "Enhanced Image": normalized_image
            }

            for idx, img in enumerate(enhanced_images):
                imgs[f"Scale {idx} Output"] = img

            display.plot_figures(imgs, 1,len(imgs)) 
        
        return normalized_image, enhanced_images, self.lse
        



