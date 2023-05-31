import tempfile
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PIL
from skimage.morphology import skeletonize
import pytesseract


def normalize_image(image):
    norm_image = np.zeros(image.shape, dtype=np.uint8)
    cv2.normalize(image, norm_image, 0, 255, cv2.NORM_MINMAX)
    return norm_image


def skew_correction(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to obtain a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (presumably the text region)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the polygonal curve of the contour
    polygon = cv2.approxPolyDP(largest_contour, 3, True)
    
    # Determine the minimum area rectangle enclosing the polygon
    rect = cv2.minAreaRect(polygon)
    angle = rect[-1]
    
    # Rotate the image to correct the skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    
    # Return the skew-corrected image
    return rotated_image

def remove_noise(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filtering for noise removal while preserving edges
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    
    return denoised_image

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    _, thresholded = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    return thresholded

def thinning_skeletonization(image):
    # Binarize the image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Perform skeletonization
    skeleton = skeletonize(binary)
    
    # Convert the skeleton image back to uint8 format
    skeleton_image = skeleton.astype(np.uint8) * 255
    
    return skeleton_image


def optimze(image, threshold_value=127, kernel_size=3, sharpen_strength=3):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Remove noise using morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Sharpen the image
    blurred = cv2.GaussianBlur(opening, (0, 0), 3)
    sharpened = cv2.addWeighted(opening, 1 + sharpen_strength, blurred, -sharpen_strength, 0)
    
    return sharpened


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

tessdata_dir_config = '--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'


image = cv2.imread('3.png')

newimage = normalize_image(image)
# simage = skew_correction(newimage)
# denoised_image = remove_noise(newimage)
gray = get_grayscale(newimage)
thresholded = thresholding(gray)
final_image = optimze(thresholded)
thinning = thinning_skeletonization(final_image)

output_path = 'enhanced_image.png'  # Provide the output file path
cv2.imwrite(output_path, thinning)

# # ocr_image = Image.open(output_path)

text = pytesseract.image_to_string(final_image)

print(text)

# cv2.imshow("Enhanced Image", newimage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
