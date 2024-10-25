import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import uuid
import os
import cv2
import numpy as np
from utils import create_range_hue, padding, mask_img, \
                    find_main_color, remove_padding, fix_path
from utils import RANGE_HUE_LABEL
import yaml
import cProfile
from time import time 
model = YOLO("yolo11x-seg.pt")
os.makedirs('output', exist_ok=True)
all_hue_range = create_range_hue()

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

HOST_IP = config['HOST_IP'] 
PORT_NUM = config['PORT_NUM'] 
AVAIL_CLASS = config["AVAIL_CLASS"]
avail_class = AVAIL_CLASS.values()




if __name__ == '__main__':
    # Profile the script
    target_class = "motorcycle"
    final_res = {target_class: [0, None]}

    img_path = 'D:\\ManhT04\\test_color\\ROI6\\2.jpg'

    img_path = fix_path(img_path)
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        print("Error: Image not loaded. Check the path.")


    uuid = str(uuid.uuid4())
    out_folder = f'output/{uuid}'
    os.makedirs(out_folder, exist_ok=True)
    print ('saved folder', out_folder)
    pad_img_path = os.path.join(out_folder, 'padded.png')
    # padding(img_path, pad_img_path)


    padded_image = padding(img_path, a = 100) # PIL RGB result

    results = model.predict(conf=0.2, source=padded_image, save=False)
    for r in results:
        img = np.copy(r.orig_img)
        for ci, c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]
            conf = c.boxes.conf.tolist().pop()
            isolated = mask_img(img=img, c=c)   
            isolated = remove_padding(isolated)
            cv2.imwrite(f'output/{label}_{ci}_{conf:.2f}.png', isolated)
            main_hue_range = find_main_color(isolated, all_hue_range) 
            rgb_class = RANGE_HUE_LABEL[main_hue_range]

            save_path_isolated = os.path.join(out_folder, f'{label}_{ci}_{conf:.2f}.png')
            cv2.imwrite(save_path_isolated, isolated) # isolated is BGR since yolo output is BGR
            # print ('save_path_isolated', save_path_isolated)
            # print ('main_hue_range', rgb_class)
            if label == target_class:
                if conf > final_res[target_class][0]:
                    final_res[target_class][0] = conf
                    final_res[target_class][1] = rgb_class
    color = final_res[target_class][1]
    print (color)
