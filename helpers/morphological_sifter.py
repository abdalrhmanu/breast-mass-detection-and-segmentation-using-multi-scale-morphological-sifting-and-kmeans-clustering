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
        pass

    # Generate Morphological Kernels for Sifting
    def generate_rotated_kernel(self, D, delta_theta):
        kernel = np.zeros((D, D), dtype=np.uint8)
        kernel[int(D/2)] = 1  # Set the middle row to 1

        rotmat = cv2.getRotationMatrix2D((int(D/2), int(D/2)), delta_theta, 1.0)
        rotated_kernel = cv2.warpAffine(kernel, rotmat, (D, D))

        non_zero_cols = np.any(rotated_kernel, axis=0)
        non_zero_rows = np.any(rotated_kernel, axis=1)
        rotated_kernel = rotated_kernel[non_zero_rows][:, non_zero_cols]

        return rotated_kernel


    # Multi-Scale Morphological Sifting
    def multi_scale_morphological_sifters(self, input_img ,M, N, Areamin, Areamax, PixelSize):
        SI = np.zeros((M+1), dtype=float) # Scale Interval
        D1 = np.zeros((M), dtype=int)     # Outer Diameter
        D2 = np.zeros((M), dtype=int)     # Inner Diameter

        # Calculate Minimum/Maximum Diameter
        DImin = 2 * math.sqrt(Areamin/math.pi)/(PixelSize*4)  
        DImax = 2 * math.sqrt(Areamax/math.pi)/(PixelSize*4)
        
        SI[0] = 1.   # Minimum Dimention in First Iteration
        
        # Calculate Diameters D1,D2
        for i in range(1,M+1):
            SI[i] = ((DImax/DImin)**(1/M))**(i)
        for i in range(0,M):
            D1[i] = np.round(DImin * SI[i])
            if ((D1[i]%2) == 0):  # Ensure Odd-Numbered Diameter to use as Kernel Filter
                D1[i]+=1;
            D2[i] = np.round(DImin * SI[i+1])
            if ((D2[i]%2) == 0):  # Ensure Odd-Numbered Diameter to use as Kernel Filter
                D2[i]+=1;
        # Placeholder for Summing Image    
        sum_all_images = np.zeros((M,input_img.shape[0],input_img.shape[1]), dtype=int)
        for i in range(0,M):
            for delta_theta in range(0,180,int(180/N)):
                
                # Apply Top-Hat Transform using Outer Diameter Line
                rotated_kernel = self.generate_rotated_kernel(D2[i], delta_theta)
                dst1           = cv2.morphologyEx(input_img, cv2.MORPH_TOPHAT, rotated_kernel)
        
                # Apply Morphological Opening using Inner Diameter Line
                rotated_kernel = self.generate_rotated_kernel(D1[i], delta_theta)
                dst2           = cv2.morphologyEx(dst1, cv2.MORPH_OPEN, rotated_kernel)
                
                sum_all_images[i,:,:] = sum_all_images[i,:,:] + dst2;
            sum_all_images[i,:,:] = (sum_all_images[i,:,:]/np.max(sum_all_images[i,:,:]))*(2**16-1) # Normalize Output Image
        return sum_all_images

