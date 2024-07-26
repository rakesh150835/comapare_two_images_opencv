from PIL import Image

# Open the PNG image
png_image = Image.open('/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted/image2.png')

# Convert the image to RGB mode
rgb_image = png_image.convert('RGB')

# Save the image as a JPG
rgb_image.save('/Users/apple/Desktop/images_for_opncv_detection/iloveimg-converted/output.jpg')
