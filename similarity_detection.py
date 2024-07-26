import cv2
import os


image1 = cv2.imread("/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted/image1.jpg")

file_path = '/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted'

for file in os.listdir(file_path):
    print(os.path.join(file_path, file))
    image = cv2.imread(os.path.join(file_path, file))
    

    hist_img1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img1[255, 255, 255] = 0 #ignore all white pixels
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2 = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img2[255, 255, 255] = 0  #ignore all white pixels
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # Find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    print(f"Similarity Score: ", round(metric_val, 2))
