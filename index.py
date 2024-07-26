from colorDescriptor import ColorDescriptor
import argparse
import glob
import cv2 as cv


# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

output = open("index.csv", "w") # open csv file to store feature vetor

imagePath = "/Users/apple/Desktop/images_for_opncv_detection/download.jpeg"
imageID = imagePath[imagePath.rfind("/") + 1:]
image = cv.imread(imagePath)

features = cd.describe(image)

# write the features to file
features = [str(f) for f in features]
output.write(f"{imageID},{','.join(features)}\n")

output.close()

