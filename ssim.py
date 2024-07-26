import cv2
import numpy as np
from skimage import metrics
# Load images
image1 = "/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted/image1.jpg"
image2 = "/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted/image2.jpg"


image1 = cv2.imread(image1)
image2 = cv2.imread(image2)

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation = cv2.INTER_AREA)

# Convert images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
print(f"SSIM Score: ", round(ssim_score[0], 2))
