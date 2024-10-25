import cv2
from utils import create_range_hue, hue_to_example_rgb, de_normalize_value
from PIL import Image
import colorsys

'''
For HSV, hue range is [0,179], saturation range is [0,255], 
and value range is [0,255]. Different software use different scales. 
So if you are comparing OpenCV values with them, you need to normalize these ranges.

'''
path = "/home/ai-ubuntu/hddnew/Manh/obj_color/images/banana.jpg"
img = cv2.imread(path)
x, y = 288, 288
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Get the pixel value at that position
pixel_value = img[y, x]  # Remember the format is (y, x) in OpenCV
print ('starttttt')
print (pixel_value)
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hsv_pixel = img[y,x]
print (hsv_pixel)
h, s, v = hsv_pixel
rgb = hue_to_example_rgb(h,s,v)
rgb_int = de_normalize_value(rgb)

print (rgb_int)

'''
test see if rgb to hsv to rgb conversion is correct or not 
'''