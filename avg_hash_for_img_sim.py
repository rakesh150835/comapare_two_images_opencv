import cv2
import numpy as np
from skimage import metrics
import imagehash

# Load images
image1 = "/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted/image1.jpg"
image2 = "/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted/image2.jpg"


image1 = cv2.imread(image1)
image2 = cv2.imread(image2)

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation = cv2.INTER_AREA)

hash0 = imagehash.average_hash(image1)
hash1 = imagehash.average_hash(image2)
cutoff = 5  # Can be changed according to what works best for your images

hashDiff = hash0 - hash1  # Finds the distance between the hashes of images
if hashDiff < cutoff:
    print('These images are similar!')
