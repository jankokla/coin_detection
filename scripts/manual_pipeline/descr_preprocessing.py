import cv2

import numpy as np

from skimage.color import *

def extract_annulus_region(img, xcenter, ycenter, inner_radius, outer_radius):

        outRoi = np.copy(img)
        
        mask1 = np.zeros_like(outRoi)
        mask2 = np.zeros_like(outRoi)
        
        mask1 = cv2.circle(mask1, (xcenter, ycenter), inner_radius, (255, 255, 255), -1)
        mask2 = cv2.circle(mask2, (xcenter, ycenter), outer_radius, (255, 255, 255), -1)
        
        mask = cv2.subtract(mask2, mask1)
        
        roi = cv2.bitwise_and(outRoi, mask)
    
        return roi #, (xcenter-outer_radius, ycenter-inner_radius)


def extract_annulus_region_by_proportion(img):

	x = img.shape[0]
	y = img.shape[1]

	xcenter = x // 2
	ycenter = y // 2

	max_outer_radius = np.min([x, y]) // 2

	outer_radius = round(0.90 * max_outer_radius)
	inner_radius = round(0.75 * max_outer_radius)

	outRoi = np.copy(img)
	mask1 = np.zeros_like(outRoi)
	mask2 = np.zeros_like(outRoi)

	mask1 = cv2.circle(mask1, (xcenter, ycenter), inner_radius, (255, 255, 255), -1)
	mask2 = cv2.circle(mask2, (xcenter, ycenter), outer_radius, (255, 255, 255), -1)

	mask = cv2.subtract(mask2, mask1)
	
	roi = cv2.bitwise_and(outRoi, mask)
	
	return roi

def extract_center_region_by_proportion(img):

	height, width = img.shape[0:-1]

	x = img.shape[0]
	y = img.shape[1]

	xcenter = x // 2
	ycenter = y // 2

	max_radius = np.min([x, y]) // 2
	radius = round(max_radius * 0.65)

	# Draw circle
	canvas = np.zeros((height, width))
	cv2.circle(canvas, (xcenter, ycenter), radius, (255,255,255), -1)
	
	# Create a copy of the input and mask input:
	imageCopy = img.copy()
	imageCopy[canvas == 0] = (0, 0, 0)

	# Crop the roi:
	x = xcenter - radius
	y = ycenter - radius
	h = 2 * radius
	w = 2 * radius

	croppedImg = imageCopy[y:y + h, x:x + w]

	return croppedImg

def extract_circular_region(img, xcenter, ycenter, radius=100):
	"""
	Extract part of an image as a circular region centred at image center
	"""

	height, width = img.shape[0:-1]

	# Draw circle
	canvas = np.zeros((height, width))
	cv2.circle(canvas, (xcenter, ycenter), radius, (255,255,255), -1)
	
	# Create a copy of the input and mask input:
	imageCopy = img.copy()
	imageCopy[canvas == 0] = (0, 0, 0)

	# Crop the roi:
	x = xcenter - radius
	y = ycenter - radius
	h = 2 * radius
	w = 2 * radius

	croppedImg = imageCopy[y:y + h, x:x + w]

	return croppedImg
""" 

def get_descriptors(img, descriptor_list):

	descriptor_values = {}

	for descriptor in descriptor_list:
		descriptor_function = getattr(d, str(descriptor))
		descriptor_values[descriptor] = descriptor_function(img)

	return descriptor_values


 """