import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove


def create_background_mask(image, threshold=200):
    """Create a background mask using simple color-based segmentation."""
    grayscale_image = image.convert("L")
    mask = np.array(grayscale_image) > threshold
    return Image.fromarray(mask.astype(np.uint8) * 255, "L")



def refine_mask(mask, dilation_kernel_size=3):
    """Refine a mask by applying dilation."""
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(np.array(mask), kernel, iterations=1)
    return Image.fromarray(dilated_mask)

def remove_background_ml(image, model="birefnet-general"):
    """Remove the background from an image using a machine learning model."""
    session = new_session(model)
    return remove(image, session=session, alpha_matting=True)
