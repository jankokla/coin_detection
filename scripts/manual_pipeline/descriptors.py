import numpy as np
import scipy.ndimage as ndi

from skimage import transform
from skimage.feature import hog

from scripts.manual_pipeline.descr_preprocessing import extract_circular_region, extract_annulus_region_by_proportion, extract_center_region_by_proportion

def descr_area(radius):
	return np.pi * radius**2

def descr_mean(img):
	"""
	Returns means of channels of an image

	(C1_mu, C2_mu, C3_mu)
	"""
	c1 = np.copy(img[:,:,0])
	c2 = np.copy(img[:,:,1])
	c3 = np.copy(img[:,:,2])

	return c1.mean(), c2.mean(), c3.mean()

def descr_std(img):
	"""
	Returns std of channels of an image

	(C1_std, C2_std, C3_std)
	"""
	c1 = np.copy(img[:,:,0])
	c2 = np.copy(img[:,:,1])
	c3 = np.copy(img[:,:,2])

	return c1.std(), c2.std(), c3.std()

def descr_mean_center(img):
	"""
	Extract part of the center to compute its color mean

	"""
	img_cropped = extract_center_region_by_proportion(img)

	return descr_mean(img_cropped)

def descr_std_center(img):
	"""
	Extract part of the center to compute its SD
	"""
	img_cropped = extract_center_region_by_proportion(img)
	
	return descr_std(img_cropped)

def descr_img_smoothness(img):
	return np.mean(np.absolute(ndi.filters.laplace(img / 255.0)))

def descr_hog(img, resize_height, resize_width, is_resize=True):
	fd, hog_image = hog(
	img,
	orientations=8,
	pixels_per_cell=(16, 16),
	cells_per_block=(1, 1),
	visualize=True,
	channel_axis=-1,
	)

	if is_resize:
		hog_image = transform.resize(hog_image,
			       (resize_height, resize_width),
			       anti_aliasing=True)
	
	return fd, hog_image

def descr_mean_annulus(img):

	roi = extract_annulus_region_by_proportion(img)

	return descr_mean(roi)

def descr_std_annulus(img):

	roi = extract_annulus_region_by_proportion(img)

	return descr_std(roi)

def descr_histogram_tuple_diff(t1, t2):
	return tuple(t1 - t2 for t1, t2 in zip(t1, t2))