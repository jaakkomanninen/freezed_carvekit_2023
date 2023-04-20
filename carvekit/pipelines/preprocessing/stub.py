import numpy as np
from PIL import Image
from loguru import logger

def equalize_histogram(image):
    """
    Apply histogram equalization to the given image and return the result.
    """
    logger.debug("Applying histogram equalization to image")
    # Convert the image to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert(mode='L')

    # Convert the image to a numpy array
    img_array = np.asarray(image)

    # Calculate the histogram of the image
    histogram_array = np.bincount(img_array.flatten(), minlength=256)

    # Normalize the histogram
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels

    # Calculate the cumulative histogram
    chistogram_array = np.cumsum(histogram_array)

    # Create a lookup table for pixel mapping
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

    # Apply the transformation to the image
    img_list = list(img_array.flatten())
    eq_img_list = [transform_map[p] for p in img_list]
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
    eq_img = Image.fromarray(eq_img_array, mode='L')

    return eq_img


class PreprocessingStub:
    """Stub for future preprocessing methods"""

    def __init__(self):
        self.preprocessing_functions = [equalize_histogram]

    def __call__(self, interface, images):
        """
        Passes data though interface.segmentation_pipeline() method

        Args:
            interface: Interface instance
            images: list of images

        Returns:
            the result of passing data through segmentation_pipeline method of interface
        """
        # Apply all preprocessing functions to each image
        preprocessed_images = []
        for image in images:
            for function in self.preprocessing_functions:
                image = function(image)
            preprocessed_images.append(image)
            logger.debug("Proprocessing done.")

        # Pass the preprocessed images through the segmentation pipeline
        return interface.segmentation_pipeline(images=preprocessed_images)
