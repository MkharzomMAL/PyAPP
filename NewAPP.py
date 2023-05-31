import PIL
import inspect
from PIL import Image 
from PIL import ImageEnhance
import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def enhance_image(image_path):
    # Load the image
    image = cv2.imread(image_path, 0)  # Read the image in grayscale mode

    # Apply adaptive thresholding to enhance the text
    enhanced_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Perform morphological operations to further enhance the text
    kernel = np.ones((3, 3), np.uint8)
    enhanced_image = cv2.erode(enhanced_image, kernel, iterations=1)
    enhanced_image = cv2.dilate(enhanced_image, kernel, iterations=1)

    # Save the enhanced image
    output_path = 'enhanced_image.jpg'
    cv2.imwrite(output_path, enhanced_image)
    print(f"Enhanced image saved as '{output_path}'.")

# enhance_image('1.jpg')



image_path = "binarize_image.png"

image = Image.open(image_path)

text = pytesseract.image_to_string(image)

print(text)
