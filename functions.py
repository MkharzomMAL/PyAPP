import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PIL


def normalize_image(image):
    normalized_img = np.zeros((image.shape[0], image.shape[1]))
    normalized_img = cv2.normalize(image, normalized_img, 0, 255, cv2.NORM_MINMAX)
    return normalized_img

def deskew_image(image):
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def scale_image(image_path):
    image = Image.open(image_path)
    length_x, width_y = image.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    resized_image = image.resize(size, Image.ANTIALIAS)
    temp_file = 'scaled_image.png'  # Provide a temporary file name
    resized_image.save(temp_file, dpi=(300, 300))
    return temp_file

def remove_noise(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    return denoised_image

def thinning(image):
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(thresholded_image, kernel, iterations=1)
    return eroded_image

def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def binarize_image(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

# Main function to enhance the image
def enhance_image_for_ocr(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Step 1: Normalize the image
    normalized_image = normalize_image(image)

    # Step 2: Deskew the image
    deskewed_image = deskew_image(normalized_image)

    # Step 3: Scale the image
    scaled_image_path = scale_image(image_path)
    scaled_image = cv2.imread(scaled_image_path)

    # Step 4: Remove noise
    denoised_image = remove_noise(scaled_image)

    # Step 5: Thinning/Skeletonization
    thinned_image = thinning(denoised_image)

    # Step 6: Convert to grayscale
    grayscale_image = convert_to_grayscale(thinned_image)

    # Step 7: Binarize the image
    binarized_image = binarize_image(grayscale_image)

    # Save the enhanced image
    output_path = 'enhanced_image.png'  # Provide the output file path
    cv2.imwrite(output_path, binarized_image)
    print(f"Enhanced image saved as '{output_path}'.")

# Provide the path to the image you want to enhance




image = cv2.imread('2.jpg')

normalized_image = normalize_image(image)
grayscale = convert_to_grayscale(normalized_image)
binimage = binarize_image(grayscale)

output_path = 'binarize_image.png'  # Provide the output file path
cv2.imwrite(output_path, binimage)