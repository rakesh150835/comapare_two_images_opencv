import cv2
import os
from skimage.metrics import structural_similarity as ssim


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # converting to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resizing to fixed size
    resized = cv2.resize(gray, (300, 300))

    return resized


def compare_images(image1, image2):
    # Compute the Structural Similarity Index (SSIM)
    score, _ = ssim(image1, image2, full=True)
    return score


def image_exists(target_image_path, current_image_path, similarity_threshold=0.9):
    # Preprocess the target  and current image
    target_image = preprocess_image(target_image_path)
    current_image = preprocess_image(current_image_path)

    # Compare the target image with the current image
    similarity_score = compare_images(target_image, current_image)
    
    if similarity_score > similarity_threshold:
        return True, similarity_score * 100
    else:
        return False, similarity_score * 100
    

t_img_path = "/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted/image1.jpg"
c_img_path = "/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted/output.jpg"
similarity_threshold = 0.9
print(image_exists(t_img_path, c_img_path, similarity_threshold))
    
