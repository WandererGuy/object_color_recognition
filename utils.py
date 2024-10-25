import uuid
import os
import cv2
import numpy as np
from PIL import Image
import colorsys
from collections import Counter
import yaml

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



def padding(image_path):
      # Load the image
  image = Image.open(image_path)
  # Define padding size for each side
  a = 100
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
 
def check_non_hue_color(pixel):
      '''
      
      so white object , influence by color light can still be labeled white, black influence can still be black 
      trade off is that , low light upon object , object color so fade , then it can mistake but those cases are extreme
      color light shine upon object will cause wrong detect too
      vhs can handle white light illuminous , not color light 
      
      '''
      upper_variance = 1 - LOWER_VARIANCE
      _,s, v = pixel
      #  case == 0 replace by <= MAX_VALUE * LOWER_VARIANCE
      '''
      independent of hue varies 
      s       | v
      0       | x-value   -> grey change from black to grey to white 
      x-value | 0         -> always black
      so 4 cases of color grey, black, white 
      '''
      if s != 0 and v <= MAX_VALUE * LOWER_VARIANCE:
        o = "black"
      elif s <= MAX_VALUE * LOWER_VARIANCE and v <= MAX_VALUE * LOWER_VARIANCE: # close to 0 
        o =  "black"
      elif s <= MAX_VALUE * LOWER_VARIANCE and v >= MAX_VALUE * LOWER_VARIANCE and v <= MAX_VALUE * upper_variance:
        o = "grey"
      elif s <= MAX_VALUE * LOWER_VARIANCE and v >= MAX_VALUE * upper_variance:
        o = "white"
      else: 
        return None 
      return "black_grey_white"
    
    
def create_rgb_black_grey_white():
      ls = []
      upper_variance = 1 - LOWER_VARIANCE

      for s in range(MAX_VALUE):
          for v in range(MAX_VALUE):
              if s != 0 and v <= MAX_VALUE * LOWER_VARIANCE:
                    ls.append((s, v))
              elif s <= MAX_VALUE * LOWER_VARIANCE and v <= MAX_VALUE * LOWER_VARIANCE: # close to 0 
                    ls.append((s, v))
              elif s <= MAX_VALUE * LOWER_VARIANCE and v >= MAX_VALUE * LOWER_VARIANCE and v <= MAX_VALUE * upper_variance:
                    ls.append((s, v))
              elif s <= MAX_VALUE * LOWER_VARIANCE and v >= MAX_VALUE * upper_variance:
                    ls.append((s, v))
      return ls
              

def find_main_color(img, all_hue_range):
    print ('start find main color')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = img.reshape((-1, 3))
    remain_color = []
    for x in pixels:
        if np.all(x != pixels[0]): # remove first pixels, like first background pixel in isolated segment image, complete black 
            non_hue_color = check_non_hue_color(x) 
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