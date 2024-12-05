import uuid
import os
import cv2
import numpy as np
from PIL import Image
import colorsys
from collections import Counter
import yaml
from time import time 


RANGE_HUE_LABEL = {str([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]): "red_to_orange", 
                   str([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]): "orange_to_yellow", 
                   str([30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]): "yellow_to_chartreuse_green", 
                   str([45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]): "chartreuse_green_to_green", 
                   str([60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]): "green_to_spring_green", 
                   str([75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]): "spring_green_to_cyan", 
                   str([90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104]): "cyan_to_azure", 
                   str([105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]): "azure_to_blue", 
                   str([120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134]): "blue_to_violet", 
                   str([135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]): "violet_to_magenta", 
                   str([150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164]): "magenta_to_rose", 
                   str([165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179]): "rose_to_red",
                   str(["black_grey_white"]): "black_grey_white"}

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
LOWER_VARIANCE = config["LOWER_VARIANCE"]
MAX_VALUE = config["MAX_VALUE"]
# upper_variance = 1 - LOWER_VARIANCE

def fix_path(path):
    path = str(path)
    new_path = path.replace('\\\\','/') 
    return new_path.replace('\\','/')

def create_range_hue():
    total = []
    t = []
    for i in range (0, 180): ## there is only 0 to 179 hue value in opencv
        # 12 hues : 6 primary and 6 secondary on color wheel -> 12 range can be labeled
        if i % 15 == 0 and i != 0:
            total.append(t)
            t = []
            t.append(i)
        elif i == 179:
            t.append(i) 
            total.append(t)
        else:
            t.append(i)
    total.append(['black_grey_white'])
    return total

def sort_dict_by_value(my_dict):
      sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
      return sorted_dict
    
def counting(my_list):
  # Count occurrences of each element
  element_count = Counter(my_list)
  # Print each element and its occurrence
  return element_count

def normalize_rgb(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
  
def de_normalize_value(color_tuple):
      ls = []
      for i in color_tuple:
        ls.append(round(i * 255))
      return tuple(ls)

def hue_to_example_rgb(h, s, v):
  '''
  For HSV, hue range is [0,179], saturation range is [0,255], 
  and value range is [0,255]. Different software use different scales. 
  So if you are comparing OpenCV values with them, you need to normalize these ranges.

  '''
  h = h/180
  s = s/256
  v = v/256
  # Convert to RGB
  r, g, b = colorsys.hsv_to_rgb(h, s, v)
  # Scaling RGB to 0-255 range
  rgb = tuple(i for i in (r, g, b))
  return rgb



def padding(image_path, a = 100):
      # Load the image
  image = Image.open(image_path)
  # Define padding size for each side
  top, bottom, left, right = a,a,a,a  # Adjust as needed
  # Choose padding color: (255, 255, 255) for white, (0, 0, 0) for black
  padding_color = (255, 255, 255)  # White padding, change to (0, 0, 0) for black
  # Calculate new image size with padding
  new_width = image.width + left + right
  new_height = image.height + top + bottom
  # Create a new image with the specified padding color
  padded_image = Image.new("RGB", (new_width, new_height), padding_color)
  # Paste the original image onto the center of the new image
  padded_image.paste(image, (left, top))
  # Save the padded image
#   padded_image.save(save_path)
  return padded_image

def fix_background(b_mask, img):
      # OPTION-1: Isolate object with black background
  # if save_path.endswith('.png'):
  #       transparent = True
  # else:
  #       transparent = False
  transparent = False
  if not transparent:
    # Create 3-channel mask
    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    # Isolate object with binary mask
    isolated = cv2.bitwise_and(mask3ch, img)
  else:
    # OPTION-2: Isolate object with transparent background (when saved as PNG)
    isolated = np.dstack([img, b_mask])
  return isolated


def mask_img(img, c):
  # Create binary mask
  b_mask = np.zeros(img.shape[:2], np.uint8)
  #  Extract contour result
  contour = c.masks.xy.pop()
  #  Changing the type
  contour = contour.astype(np.int32)
  #  Reshaping
  contour = contour.reshape(-1, 1, 2)
  # Draw contour onto mask
  _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
  isolated = fix_background(b_mask, img)
  return isolated
 

def remove_padding(padded_image, padding_size=100):
    """
    Removes padding from a padded image.

    :param padded_image: Padded Image object.
    :param padding_size: Size of the padding to remove from each side (default is 100).
    :return: Original Image object without padding.
    """
    # Get the size of the padded image
    # Get the dimensions of the padded image
    # Get the dimensions of the padded image
    height, width = padded_image.shape[:2]

    # Define the bounding box to crop
    left = padding_size
    top = padding_size
    right = width - padding_size
    bottom = height - padding_size

    # Crop the image to remove padding
    original_image = padded_image[top:bottom, left:right]

    return original_image
    

def check_non_hue_color(pixel, adaptive_constant):
      '''
      
      so white object , influence by color light can still be labeled white, black influence can still be black 
      trade off is that , low light upon object , object color so fade , then it can mistake but those cases are extreme
      color light shine upon object will cause wrong detect too
      vhs can handle white light illuminous , not color light 
      
      '''
      lower_variance = LOWER_VARIANCE + adaptive_constant
      # lower_limit = MAX_VALUE * lower_variance
      lower_limit = lower_variance

      _,s, v = pixel
      #  case == 0 replace by <= MAX_VALUE * LOWER_VARIANCE
      '''
      independent of hue varies 
      s       | v
      0       | x-value   -> grey change from black to grey to white 
      x-value | 0         -> always black



      so 4 cases of color grey, black, white 
      elif s <= MAX_VALUE * LOWER_VARIANCE and v <= MAX_VALUE * LOWER_VARIANCE: # close to 0 
        o =  "black"
      elif s <= MAX_VALUE * LOWER_VARIANCE and v >= MAX_VALUE * LOWER_VARIANCE and v <= MAX_VALUE * upper_variance:
        o = "grey"
      elif s <= MAX_VALUE * LOWER_VARIANCE and v >= MAX_VALUE * upper_variance:
        o = "white"
      '''
      if s <= lower_limit:
          return "black_grey_white"
      # elif s <= lower_limit and v <= lower_limit:
      #     return "black_grey_white"
      else: 
          return None 
                  
def find_main_color(img, all_hue_range):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = img.reshape((-1, 3))
    remain_color = []
    first_pixel = [0, 0, 0] # BACKGROUND IS BLACK
    remain_pixels = []
    for x in pixels:
        x = x.tolist()
        if x == first_pixel:
            continue
        else: 
            remain_pixels.append(x)

    adaptive_constant = 10

    for x in remain_pixels:
        non_hue_color = check_non_hue_color(pixel = x, adaptive_constant=adaptive_constant) 
        
        if non_hue_color != None:
          remain_color.append(non_hue_color) # check if in black_grey_white first 
        else:
          remain_color.append(x[0]) 
    # remain_color = [x[0] for x in pixels if np.all(x != pixels[0])] # remove black pixels, like first padding pixel
    # keep the hue only 
    # hue class range from 0 to 179 in opencv 
    single_hue_count = counting(remain_color) # count occurence of each hue value 
    t = {}
    for hue_range in all_hue_range:
          t[str(hue_range)] = 0
          for hue_value in hue_range:
                t[str(hue_range)] += single_hue_count[hue_value]
    hue_range_sorted_dict = sort_dict_by_value(t)
    main_hue_range = next(iter(hue_range_sorted_dict))
    return main_hue_range

def pseudo_find_main_color(img, all_hue_range, add):
    print ('start find main color')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = img.reshape((-1, 3))
    remain_color = []
    remain_pixels = []
    remain_lightness = []
    remain_saturation = []
    remain_hue = []
    first_pixel = [0, 0, 0] # BACKGROUND IS BLACK, this might also delete pure black object
    for x in pixels:
        x = x.tolist()
        if x == first_pixel:
            continue
        else: 
            remain_pixels.append(x)
            remain_lightness.append(x[2])
            remain_saturation.append(x[1])
            remain_hue.append(x[0])
    average_lightness = cal_average(remain_lightness)
    average_saturation = cal_average(remain_saturation)
    average_hue = cal_average(remain_hue)
    adaptive_constant = cal_adapt_variance(add, average_lightness)
    for x in remain_pixels:
        non_hue_color = check_non_hue_color(pixel = x, adaptive_constant=adaptive_constant) 
        
        if non_hue_color != None:
          remain_color.append(non_hue_color) # check if in black_grey_white first 
        else:
          remain_color.append(x[0]) 
    # remain_color = [x[0] for x in pixels if np.all(x != pixels[0])] # remove black pixels, like first padding pixel
    # keep the hue only 
    # hue class range from 0 to 179 in opencv 
    single_hue_count = counting(remain_color) # count occurence of each hue value 
    t = {}
    for hue_range in all_hue_range:
          t[str(hue_range)] = 0
          for hue_value in hue_range:
                t[str(hue_range)] += single_hue_count[hue_value]
    hue_range_sorted_dict = sort_dict_by_value(t)
    main_hue_range = next(iter(hue_range_sorted_dict))
    return main_hue_range, average_hue, average_saturation, average_lightness


def cal_adapt_variance(add, average_lightness):
    # light range from 0 to 255 in opencv HSV 
    print ("average_lightness", average_lightness)
    '''
    while black lower or upper saturation , it is still black , 
    white or gray still fsall in class black_grey_white, 
    however illumation/ light color 's light saturation effect will be filtered out 
    (by adaptive_constant), so that its saturation need be meet higher limit be escape 
    black_grey_white class (like black hole lol)
    with 179/255 lightness , we need to raise limit for LOWER_VARIANCE 
    but little lightness or shadow, we need decrease saturation , or else
    it falls to black_grey_white class
    hence, i introduce to u the adaptive_constant

    
    '''
    return add
  
def cal_average(ls): # calculate object light upon , since light color can affect object color
    # light range from 0 to 255 in opencv HSV 
    average = sum(ls) / len(ls)
    return average


# def find_average_color_rgb(save_path, img, n_main_color = 1):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
#     print ('start find main color')
#     img = normalize_rgb(img)
#     pixels = img.reshape((-1, 3))
#     print (type(pixels[0]))
#     print (pixels[0])
#     remain_color = tuple(x for x in pixels if np.all(x != pixels[0])) # remove black pixels, like first padding pixel
#     clt = KMeans(n_clusters=n_main_color)
#     clt.fit(remain_color)
#     centers = clt.cluster_centers_
#     rgb = [[i for i in center] for center in centers] # get all centroid values
#     cv2.imwrite(save_path, img)
#     return rgb[0]