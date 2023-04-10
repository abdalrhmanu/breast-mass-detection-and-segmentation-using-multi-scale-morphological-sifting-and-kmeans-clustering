import cv2
from loguru import logger
import display

class Preprocessor:
    def __init__(self):
        self.scale_factor       = 0.4

        self._resized_img           = None
        self._gray_img              = None
        self._thresholding_mask     = None
        self._contour_img           = None
        self._segmented_img         = None
        self._rescaled_img          = None
        self._clahe_img             = None


    def _resize(self, images):
        if isinstance(images, list):
            # implement this block to handle passing a list of images
            return
        logger.info(f"Resizing with a scale factor {self.scale_factor} INTER_CUBIC interpolation.")
        return cv2.resize(images.copy(), None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_CUBIC)
    
    def _to_grayscale(self, images):
        if isinstance(images, list):
            # implement this block to handle passing a list of images
            return

        return cv2.cvtColor(images.copy(), cv2.COLOR_BGR2GRAY)        
    
    def _threshold_mask(self, images):
        if isinstance(images, list):
            # implement this block to handle passing a list of images
            return

        return cv2.threshold(images.copy(), 30, 255, cv2.THRESH_BINARY)[1]
    
    def _find_contours_and_segment(self, images):
        if isinstance(images, list):
            # implement this block to handle passing a list of images
            return
        
        contours, _ = cv2.findContours(self._thresholding_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        contour_image = cv2.drawContours(images.copy(), [largest_contour], -1, (255, 255, 255), 3)
        cropped_image = images.copy()[y:y+h, x:x+w]

        return contour_image, cropped_image
    
    def _rescale(self, images):
        if isinstance(images, list):
            # implement this block to handle passing a list of images
            return
        
        return cv2.normalize(images.copy(), None, 0, 255, cv2.NORM_MINMAX)
    
    def _clahe(self, images):
        if isinstance(images, list):
            # implement this block to handle passing a list of images
            return
        
        clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(4,4))
        return clahe.apply(images.copy())
    
    
    def fit(self, images, plot = False):
        self._resized_img = self._resize(images)
        self._gray_img = self._to_grayscale(self._resized_img)
        self._thresholding_mask = self._threshold_mask(self._gray_img)
        self._contour_img, self._segmented_img = self._find_contours_and_segment(self._gray_img)
        self._rescaled_img = self._rescale(self._segmented_img)
        self._clahe_img = self._clahe(self._rescaled_img)


        # Some loggers to help keep track of the process
        logger.info(f"Original image shape: {images.shape}")
        logger.info(f"Resized image shape: {self._resized_img.shape}.")
        logger.info(f"Grayscale image shape: {self._gray_img.shape}")  
        logger.info(f"Thresholded Mask Shape: {self._thresholding_mask.shape}")       

        # Displaying some results for validation
        if plot:
            imgs = {
                "Original": images, 
            #     "Threshold Mask": thresholding_mask, 
            #     "Contour on Gray Image": contour_image, 
            #     "Cropped Image (Grayscale Version)": cropped_image, 
            #     "Rescaled 16-bit":rescaled_img,
                "Preprocessed (rescaled, pre-segmented, and enhanced)": self._clahe_img
            }

            display.plot_figures(imgs, 1,2) 

        return  self._resized_img, \
                self._gray_img, \
                self._thresholding_mask, \
                self._contour_img, \
                self._segmented_img, \
                self._rescaled_img, \
                self._clahe_img



