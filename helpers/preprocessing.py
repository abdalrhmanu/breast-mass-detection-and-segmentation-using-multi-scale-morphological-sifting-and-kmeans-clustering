import cv2
from loguru import logger
import display
import glob
import os
import time
from tqdm import tqdm, tqdm_notebook
import numpy as np

from display import convert_to_16bit

class Preprocessor:
    def __init__(self):
        self._resized_img           = None
        self._gray_img              = None
        self._thresholding_mask     = None
        self._contour_img           = None
        self._segmented_img         = None
        self._rescaled_img          = None
        self._clahe_img             = None
        self.scale_factor           = 10


    def _resize(self, images):
        # logger.info(f"Resizing with a scale factor {self.scale_factor} INTER_CUBIC interpolation.")
        return cv2.resize(images.copy(), None, fx=1/self.scale_factor, fy=1/self.scale_factor, interpolation=cv2.INTER_CUBIC)
    
    def _to_grayscale(self, images):
        return cv2.cvtColor(images.copy(), cv2.COLOR_BGR2GRAY)        
    
    def _threshold_mask(self, images):
        return cv2.threshold(images.copy(), 30, 255, cv2.THRESH_BINARY)[1]
    
    def _find_contours_and_segment(self, images):
        contours, _ = cv2.findContours(self._thresholding_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        contour_image = cv2.drawContours(images.copy(), [largest_contour], -1, (255, 255, 255), 3)
        cropped_image = images.copy()[y:y+h, x:x+w]

        return contour_image, cropped_image
    
    def _rescale(self, images):       
        return cv2.normalize(images.copy(), None, 0, 255, cv2.NORM_MINMAX)
    
    def _clahe(self, images):
        clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(4,4))
        return clahe.apply(images.copy())
    
    def _export_processed(self, dir, img_processed, img_path):
        folder_directory = os.path.join('..', dir.split('\\')[1], 'processed', dir.split('\\')[2])
        file_directory = os.path.join(folder_directory, img_path.split('\\')[-1])

        if not os.path.isdir(folder_directory):
            os.makedirs(folder_directory)
            logger.info(f"New directory created '{folder_directory}'")

        cv2.imwrite(file_directory ,img_processed)

    def _flip_breast(self, img):
        # Determine the orientation of the breast and flip the image if required
        rotation_threshold = np.median(img[:1000, :400])

        if rotation_threshold < 10:
            img = cv2.flip(img, 1)
        
        return img
    
    def _add_padding(self, img, path, ratio=1):
        length, width = img.shape[:2]
        if length / width > ratio:
            add_wid = round(length / ratio - width)
            pad = np.zeros((length, add_wid), dtype=img.dtype)
            if '_R_' in path:
                return np.concatenate((pad, img), axis=1)
            return np.concatenate((img, pad), axis=1)
        return img

        
    def fit(self, dataset_path, process_n = None, plot = False, export_processed = False):
        if isinstance(dataset_path, str):
            start_time = time.time()
            logger.info("Started processing pipeline.")

            full_path_dirs = glob.glob(dataset_path+"\\*.tif")

            if process_n is None:
                process_n = len(full_path_dirs)
            elif process_n == 0:
                logger.warning("The number of processed images process_n can't be 0. \
                               To process the entire dataset, remove the argument when calling the function.")
                return
                        
            for path in tqdm(full_path_dirs[:process_n]):
                img = cv2.imread(path)
                self._gray_img = self._to_grayscale(img)
                self._resized_img = self._resize(self._gray_img)
                self._thresholding_mask = self._threshold_mask(self._gray_img)
                self._contour_img, self._segmented_img = self._find_contours_and_segment(self._gray_img)
                self._rescaled_img = self._rescale(self._segmented_img)
                self._clahe_img = self._clahe(self._rescaled_img)
                
                # Some loggers to help keep track of the process
                # logger.info(f"Original image shape: {img.shape}")
                # logger.info(f"Resized image shape: {self._resized_img.shape}.")
                # logger.info(f"Grayscale image shape: {self._gray_img.shape}")  
                # logger.info(f"Thresholded Mask Shape: {self._thresholding_mask.shape}")       
                # logger.info(f"Preprocessing pipeline complete.")       
                
                # Displaying some results for validation
                if plot:
                    imgs = {
                        "Original": img, 
                    #     "Threshold Mask": thresholding_mask, 
                    #     "Contour on Gray Image": contour_image, 
                    #     "Cropped Image (Grayscale Version)": cropped_image, 
                    #     "Rescaled 16-bit":rescaled_img,
                        "Preprocessed": self._rescaled_img
                    }

                    display.plot_figures(imgs, 1,2) 

                
                # Export processed images
                if export_processed:                
                    self._export_processed(dataset_path, self._rescaled_img, path)

            logger.info(f"Finished processing {len(full_path_dirs)} files in approximately {(time.time() - start_time):.03f} seconds.")
        else:
            raise NotImplementedError("dataset_path must be a directory string to the dataset files. Other formats are not yet implemented.")

        return



