import cv2
import numpy as np
from PIL import Image


def create_background_mask(image: Image.Image, threshold: int = 200) -> Image.Image:
    """
    Create a background mask using simple color-based segmentation (thresholding).
    
    Args:
        image: The input PIL Image.
        threshold: Luminance threshold (0-255). Values above this are considered background.
        
    Returns:
        A black and white PIL Image mask (255 for background, 0 for subject).
    """
    grayscale_image = image.convert("L")
    mask = np.array(grayscale_image) > threshold
    return Image.fromarray(mask.astype(np.uint8) * 255, "L")


def refine_mask(mask: Image.Image, dilation_kernel_size: int = 3) -> Image.Image:
    """
    Refine a mask by applying dilation to slightly expand the ignored background area.
    
    Args:
        mask: The input mask as a PIL Image.
        dilation_kernel_size: Size of the dilation kernel.
        
    Returns:
        The refined mask as a PIL Image.
    """
    if dilation_kernel_size <= 0:
        return mask
        
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(np.array(mask), kernel, iterations=1)
    return Image.fromarray(dilated_mask)


def remove_background_ml(image: Image.Image, model: str = "u2net") -> Image.Image:
    """
    Remove the background from an image using a machine learning model.
    
    Args:
        image: The input PIL Image.
        model: The name of the rembg model to use.
        
    Returns:
        The image with background removed (RGBA).
    """
    from rembg import new_session, remove  # heavy import, only needed here

    return remove(image, session=new_session(model))

