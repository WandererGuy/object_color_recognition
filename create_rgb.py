from PIL import Image
import colorsys
from utils import create_range_hue, hue_to_example_rgb, de_normalize_value
from utils import RANGE_HUE_LABEL
import os
all_sample_rgb_range = {}
hue_range_ls = create_range_hue()
folder_rgb_for_hue = "rgb_for_hue_range"

'''
For HSV, hue range is [0,179], saturation range is [0,255], 
and value range is [0,255]. Different software use different scales. 
So if you are comparing OpenCV values with them, you need to normalize these ranges.

'''

# Define the number of steps for saturation and value
saturation_steps = 18  # From 0 to 255 in increments of 15 for 17 times
value_steps = 18      # From 0 to 255 in increments of 15 for 17 times
# every 0, 15, ... 255 (17 times step 15 )

for idx, hue_range in enumerate(hue_range_ls):
    if hue_range[0] == 'black_grey_white':
        continue 
    print (hue_range)
    hue_list = hue_range
    width = len(hue_list) * saturation_steps
    height = value_steps
    # Create a new image for the hue range
    img = Image.new('RGB', (width, height))

    # Loop over saturation levels
    for s_idx in range(saturation_steps):
        s = s_idx * 15  # Saturation from 0% to 100%
        # Loop over hue values within the hue range
        for x_idx, hue in enumerate(hue_list):
            x = x_idx + s_idx * len(hue_list)
            for v_idx in range(value_steps):
                v = v_idx * 15  # Value from 0% to 100%
                # 1 image is a square rgb value from hsv value (hue, s, v)
                rgb = hue_to_example_rgb(hue,s,v)
                rgb_int = de_normalize_value(rgb)
                r_int, g_int, b_int = rgb_int
                y = height - v_idx - 1
                # Place the pixel in the image
                img.putpixel((x, y), (r_int, g_int, b_int)) 
    # Save the image for the hue range
    os.makedirs(folder_rgb_for_hue, exist_ok=True)
    label = RANGE_HUE_LABEL[str(hue_range)] 
    img.save(f'{folder_rgb_for_hue}/{idx}_{label}.png')


