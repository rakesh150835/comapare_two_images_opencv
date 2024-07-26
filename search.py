from colorDescriptor import ColorDescriptor
from searcher import Searcher
import argparse
import cv2 as cv

# for similar image
query_image_path = "/Users/apple/Desktop/images_for_opncv_detection/target_image.jpeg"

# for completely different image 
index_csv_file = "index.csv"

cd = ColorDescriptor((8, 12, 3))

query = cv.imread(query_image_path)
features = cd.describe(query)

searcher = Searcher(index_csv_file)
results = searcher.search(features)

for value in results.values():
    print("Similarity score is: ", value)