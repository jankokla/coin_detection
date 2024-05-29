
import numpy as np
from PIL import Image, ImageEnhance
from typing import List
from skimage import io, transform, color, filters
from skimage.color import rgb2hsv
import cv2 as cv
from sklearn.metrics import pairwise_distances
from skimage.morphology import disk, closing, opening, black_tophat, white_tophat, erosion, dilation
from skimage.morphology import remove_small_holes, remove_small_objects


def apply_operation(img, operation_list, operation_dict):

    output_list = []

    input_img = np.copy(img)

    for operation in operation_list:
        output_img = np.copy(operation_dict[operation](input_img))
        output_list.append(output_img)
        input_img = np.copy(output_img)
    
    return output_list

def remove_holes(img_th, size=15):
    """
    Remove holes from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of holes

    Return
    ------
    img_holes: np.ndarray (M, N)
        Image after remove holes operation
    """
    return remove_small_holes(img_th, size)


def remove_objects(img_th, size):
    """
    Remove objects from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of objects

    Return
    ------
    img_obj: np.ndarray (M, N)
        Image after remove small objects operation
    """
    return remove_small_objects(img_th, size)

def apply_black_tophat(img, disk_size=15):

    footprint = disk(disk_size)

    img_blackth = np.zeros_like(img)

    for channel in range(img.shape[2]):
        img_blackth[:,:,channel] = black_tophat(img[:,:,channel], footprint)

    return img_blackth

def apply_white_tophat(img, disk_size=15):

    footprint = disk(disk_size)

    img_whiteth = np.zeros_like(img)

    for channel in range(img.shape[2]):
        img_whiteth[:,:,channel] = white_tophat(img[:,:,channel], footprint)

    return img_whiteth

def apply_dilation(img, disk_size=15):
    """
    Apply dilation to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N, C)
        Image mask of size MxNxC.
    disk_size: int
        Size of the disk to use for dilation

    Return
    ------
    img_closing: np.ndarray (M, N, C)
        Image after dilation operation
    """
    footprint = disk(disk_size)

    img_dilation = np.zeros_like(img)

    for channel in range(img.shape[2]):
        img_dilation[:,:,channel] = dilation(img[:,:,channel], footprint)

    return img_dilation

def apply_erosion(img, disk_size=15):
    """
    Apply erosion to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N, C)
        Image mask of size MxNxC.
    disk_size: int
        Size of the disk to use for erosion

    Return
    ------
    img_closing: np.ndarray (M, N, C)
        Image after erosion operation
    """
    footprint = disk(disk_size)

    img_erosion = np.zeros_like(img)

    for channel in range(img.shape[2]):
        img_erosion[:,:,channel] = erosion(img[:,:,channel], footprint)

    return img_erosion

def apply_closing(img_th, disk_size):
    """
    Apply closing to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for closing

    Return
    ------
    img_closing: np.ndarray (M, N)
        Image after closing operation
    """
    footprint = disk(disk_size)
    img_open = np.zeros_like(img_th)

    for channel in range(img_th.shape[2]):
        img_open[:,:,channel] = closing(img_th[:,:,channel], footprint)

    return img_open


def apply_opening(img_th, disk_size=15):
    """
    Apply opening to input mask image using disk shape ON EACH CHANNEL

    Args
    ----
    img_th: np.ndarray (M, N, C)
        Image mask of size MxNxC.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_opening: np.ndarray (M, N, C)
        Image after opening operation
    """

    footprint = disk(disk_size)
    img_open = np.zeros_like(img_th)

    for channel in range(img_th.shape[2]):
        img_open[:,:,channel] = opening(img_th[:,:,channel], footprint)

    return img_open


def read_image(
        filename: str,
        new_height: int = 400,
        new_width: int = 600
) -> np.ndarray:
    """
    Helper function for reading image and resizing it.

    Args:
        filename (str): only filename w.o. any other path
        new_height (int): in pixels
        new_width (int): in pixels

    Returns:
        resized_image (np.ndarray): values in [0, 255]
    """
    image = io.imread(f'data/train/{filename}')
    resized = transform.resize(image, (new_height, new_width), anti_aliasing=True)

    return (resized * 255).astype(np.uint8)

def read_image_from_path(
        filename: str,
        new_height: int = 400,
        new_width: int = 600,
        is_resize: bool = False
) -> np.ndarray:
    """
    Helper function for reading image and resizing it.

    Args:
        filename (str): filename
        new_height (int): in pixels
        new_width (int): in pixels

    Returns:
        resized_image (np.ndarray): values in [0, 255]
    """
    image = io.imread(filename)
    

    if is_resize:
        output = transform.resize(image, (new_height, new_width), anti_aliasing=True)
    else:
        output = image

    return (output * 255).astype(np.uint8)

def _filter_between_lines(
        hue_image: np.ndarray,
        saturation_image: np.ndarray
) -> np.ndarray:
    """
    Helper function for applying threshold with two linear separators.

    Args:
        hue_image (np.ndarray): with shape (H, W)
        saturation_image (np.ndarray): with shape (H, W)

    Returns:
        mask (np.ndarray):
    """
    # TODO: move them to some config
    # define the parameters for the lines
    slope1, slope2 = 8.9, 3
    intercept1, intercept2 = 0.2, 0.7

    # define lines
    y_line1 = slope1 * hue_image.astype(np.float32) + intercept1
    y_line2 = slope2 * hue_image.astype(np.float32) + intercept2

    # create a mask for pixels between the two lines
    between_mask = ((saturation_image > np.minimum(y_line1, y_line2))
                    & (saturation_image < np.maximum(y_line1, y_line2)))

    mask = np.where(between_mask, 0, 1)

    return mask


def apply_hsv_threshold(img):
    """
    Apply threshold to the input image in hsv colorspace.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    img_th: np.ndarray (M, N)
        thresholded image.
    """
    # Use the previous function to extract HSV channels
    data_h, data_s, data_v = extract_hsv_channels(img=img)

    mask = _filter_between_lines(data_h, data_s)

    hue_thresh = 0.3
    value_thresh = 0.95

    mask = np.logical_and(data_h < hue_thresh, mask)
    mask = np.logical_or(mask, value_thresh > 0.95)

    img[~mask] = (255, 255, 255)

    return img


def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """
    img_hsv = rgb2hsv(np.copy(img))
    data_h = img_hsv[:, :, 0]
    data_s = img_hsv[:, :, 1]
    data_v = img_hsv[:, :, 2]

    return data_h, data_s, data_v


def adjust_contrast(image: np.ndarray, contrast: float = 1.9) -> np.ndarray:
    """
    Return an image that has contrast adjusted.
    Args:
        image (np.ndarray): with shape (H, W, C)
        contrast (float): multiplier of contrast

    Returns:
        enhanced (np.ndarray): image with adjusted contrast
    """
    pil_img = Image.fromarray(image)
    enhanced = ImageEnhance.Contrast(pil_img).enhance(contrast)

    return np.array(enhanced)


def apply_sobel(image: np.ndarray) -> np.ndarray:
    """
    Return image convoluted with sobel filter.

    Args:
        image (np.ndarray): with shape (H, W, C)

    Returns:
        sobel (np.ndarray): convoluted image
    """
    grayscale = color.rgb2gray(image)
    sobel = filters.sobel(grayscale)
    return (sobel * 255).astype(np.uint8)

def get_hough_circles(
        image: np.ndarray,
        plot_image,
        is_filter: bool = True,
        param_1: int = 200,
        param_2: int = 30,
        min_radius: int = 18,
        max_radius: int = 55
):
    """
    Return Transformed circles without image

    Args:
        image (np.ndarray): with shape (H, W, C)
        plot_image (np.ndarray): with shape (H, W, C)
        is_filter (bool): if filtering of circles is applied
        param_1 (int): upper threshold
        param_2 (int): lower threshold
        min_radius (int): of Hough circle
        max_radius (int): of Hough circle

    Returns:
        hough_circles (List): List of Hough circles
    """
    circles = cv.HoughCircles(
        image, cv.HOUGH_GRADIENT, 1, 20, param1=param_1,
        param2=param_2, minRadius=min_radius, maxRadius=max_radius
    )
    if is_filter:
        circles = filter_circles(circles)
    else:
        circles = np.uint16(np.around(circles))[0, :]

    return circles

def scale_hough_circles(hough_circles, ratio, max_width, max_height):

    # Rescale circles
    scaled_circles = (hough_circles * ratio).astype(int)
    
    # TODO: Check that circle doesn't go outside of picture
    #scaled_circles[scaled_circles[:,0] + scaled_circles[:,2] > max_width ] = max_width
    #scaled_circles[scaled_circles[:,1] + scaled_circles[:,2] > max_height ] = max_height

    return scaled_circles

def apply_hough(
        image: np.ndarray,
        plot_image: np.ndarray,
        is_filter: bool = True,
        param_1: int = 200,
        param_2: int = 30,
        min_radius: int = 18,
        max_radius: int = 55
) -> np.ndarray:
    """
    Return image with Hough Transformed circles.

    Args:
        image (np.ndarray): with shape (H, W, C)
        plot_image (np.ndarray): with shape (H, W, C)
        is_filter (bool): if filtering of circles is applied
        param_1 (int): upper threshold
        param_2 (int): lower threshold
        min_radius (int): of Hough circle
        max_radius (int): of Hough circle

    Returns:
        hough_img (np.ndarray): image with Hough circles
    """
    circles = cv.HoughCircles(
        image, cv.HOUGH_GRADIENT, 1, 20, param1=param_1,
        param2=param_2, minRadius=min_radius, maxRadius=max_radius
    )
    if is_filter:
        circles = filter_circles(circles)
    else:
        circles = np.uint16(np.around(circles))[0, :]

    hough_img = plot_image.copy()

    for (x, y, r) in circles:
        cv.circle(hough_img, (x, y), r, (0, 255, 0), 4)

    return hough_img


def filter_circles(hough_output: np.ndarray) -> np.ndarray:
    """
    If Hough Transform returns circles that overlap each other,
        filter them out and keep only the biggest circle to make
        sure that the coin is fully covered.

    Args:
        hough_output (np.ndarray): with shape (1, N, 3) or shape(N, 3)

    Returns:
        filtered_output (np.ndarray): with shape (K, 3)
    """
    # Check for None
    if hough_output is not None:
        # if shape is not (N, 3) make it so
        if len(hough_output.shape) != 2:
            hough_output = hough_output.squeeze(0)

        # make sure you have uint16 as dtype for cropping and plotting
        if hough_output.dtype != np.dtype('uint16'):
            hough_output = np.uint16(np.around(hough_output))

        # extract centers and radii
        centers, radii = hough_output[:, :2], hough_output[:, 2]

        distances = pairwise_distances(centers[:, :2])

        # get call the overlapping circles and clean bottom half
        is_inside = distances < radii
        is_inside[np.tril_indices(len(is_inside), 0)] = False

        # iterate over indices to find what to keep
        keep = np.full(len(hough_output), True)
        for i in range(len(hough_output)):

            if not keep[i]:
                continue

            # find all circles where i's center is inside and i is not the largest
            overlapping = is_inside[:, i]
            larger = radii[i] > radii[overlapping]
            if not all(larger):
                keep[i] = False

            # keep only the biggest circle
            keep[overlapping & (radii[i] >= radii)] = False

        return hough_output[keep]
    
    else:
        return np.full(shape=(1,3), fill_value=1)
